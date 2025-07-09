import pandas as pd
import numpy as np
import joblib
import json
from scipy.optimize import linprog
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns  # 虽然在这个版本中没有直接用seaborn的新图，但保留导入以备后续可能的美化

# --- 0. 定义常量与加载配置 ---
print("--- 阶段五：基于CVaR的投资组合优化与组合/单资产风险度量 ---")

MAIN_ASSETS_KEYS = ['etf', 'gbi', 'cea']
RETURN_COL_NAMES = {
    'etf': 'r_etf',
    'gbi': 'r_gbi',
    'cea': 'r_cea'
}
ML_ASSETS = ['etf', 'cea']
EXPECTED_ML_FEATURE_COLS = ['lag1_r_j', 'lag3_r_j', 'lag5_r_j', 'roll_mean5_r_j',
                            'lag1_r_fu', 'lag3_r_fu', 'lag5_r_fu', 'roll_mean5_r_fu',
                            'lag1_r_jm', 'lag3_r_jm', 'lag5_r_jm', 'roll_mean5_r_jm']
WEIGHT_COL_NAMES = {  # 确保这些键与 MAIN_ASSETS_KEYS 对应
    'etf': 'w_etf',
    'gbi': 'w_gbi',
    'cea': 'w_cea'
}
TRADING_DAYS_PER_YEAR = 252
CVAR_ALPHA = 0.95
CVAR_EPSILON_TAIL = 1 - CVAR_ALPHA
PORTFOLIO_TARGET_ANNUAL_RETURN = 0.08  # 年化目标收益率 (如您上次运行所设)

# --- 1. 加载所需数据 ---
print("\n--- 步骤 1: 加载数据 ---")
try:
    simulation_data = joblib.load('simulation_results_multi_asset.joblib')
    paths_by_date_str_keys = simulation_data['paths_by_date']
    paths_by_date = {pd.to_datetime(k): v for k, v in paths_by_date_str_keys.items()}
    asset_order_in_paths = simulation_data['asset_order']  # 这个顺序很重要
    H_HORIZON_DAYS = simulation_data['horizon_days']
    N_SIMULATIONS = simulation_data['n_simulations']
    simulation_dates_str = simulation_data['simulation_dates']
    SIMULATION_DATES = sorted([pd.to_datetime(d) for d in simulation_dates_str])
    print("多资产仿真结果 (simulation_results_multi_asset.joblib) 加载成功。")
except FileNotFoundError:
    print("错误: simulation_results_multi_asset.joblib 未找到。请先运行阶段四。")
    exit()

try:
    final_df_full = pd.read_csv("final_data_for_modeling_multi_asset.csv", index_col='date', parse_dates=True)
    print("final_data_for_modeling_multi_asset.csv (含价格和特征) 加载成功。")
except FileNotFoundError:
    print("错误: final_data_for_modeling_multi_asset.csv 未找到。请先运行阶段一和阶段三。")
    exit()

try:
    with open('all_assets_base_params.json', 'r') as f:
        all_base_params = json.load(f)
    print("所有资产的基础静态参数 (all_assets_base_params.json) 加载成功。")
except FileNotFoundError:
    print("错误: all_assets_base_params.json 文件未找到。请先运行阶段二。")
    exit()

ml_models_pipelines = {}
ml_feature_names = None
for asset_key in ML_ASSETS:
    model_filename = f'dynamic_mu_{asset_key}_model.joblib'
    try:
        model_artifacts = joblib.load(model_filename)
        ml_models_pipelines[asset_key] = model_artifacts['pipeline']
        if ml_feature_names is None:  # 从第一个加载的ML模型获取特征名
            ml_feature_names = model_artifacts.get('feature_names', EXPECTED_ML_FEATURE_COLS)
        print(f"资产 '{asset_key.upper()}' 的ML模型 ({model_filename}) 加载成功。")
    except FileNotFoundError:
        print(f"警告: 资产 '{asset_key.upper()}' 的ML模型 {model_filename} 未找到。该资产将仅使用静态mu计算预期收益。")
    except Exception as e:
        print(f"加载 {model_filename} 时发生错误: {e}")

if ml_feature_names is None:  # 如果所有ML模型都加载失败，或模型文件中没有特征名
    print("警告: 未能从任何ML模型文件中加载特征名列表，将使用预定义的特征列表。")
    ml_feature_names = EXPECTED_ML_FEATURE_COLS

ACTUAL_FUTURES_END_DATE_ADJ_STR = all_base_params.get('ACTUAL_FUTURES_END_DATE_ADJ_STR', '2024-09-05')
LAST_REAL_FUTURES_DATA_DATE_FOR_MU = pd.to_datetime(ACTUAL_FUTURES_END_DATE_ADJ_STR)
last_dynamic_mu_preds_for_E = {key: None for key in MAIN_ASSETS_KEYS}  # 用于存储各资产最新的mu_dyn预测值


# --- 2. 定义辅助函数 ---
def calculate_expected_period_return(asset_key_calc, current_date_calc,
                                     static_params_calc, ml_pipeline_calc,
                                     all_features_df_calc, trained_feature_names_list_calc,
                                     last_pred_mu_calc, horizon_days_calc):
    mu_ann = static_params_calc.get('mu_base', 0)
    if asset_key_calc in ml_models_pipelines and ml_pipeline_calc is not None:
        if current_date_calc <= LAST_REAL_FUTURES_DATA_DATE_FOR_MU:
            if current_date_calc in all_features_df_calc.index:
                features_row = all_features_df_calc.loc[[current_date_calc], trained_feature_names_list_calc]
                if not features_row.empty and not features_row.isnull().values.any().any():
                    try:
                        pred_mu_ann = ml_pipeline_calc.predict(features_row)[0]
                        mu_ann = pred_mu_ann
                        last_pred_mu_calc[asset_key_calc] = mu_ann
                    except Exception:
                        if last_pred_mu_calc.get(asset_key_calc) is not None:
                            mu_ann = last_pred_mu_calc[asset_key_calc]
                elif last_pred_mu_calc.get(asset_key_calc) is not None:
                    mu_ann = last_pred_mu_calc[asset_key_calc]
            elif last_pred_mu_calc.get(asset_key_calc) is not None:
                mu_ann = last_pred_mu_calc[asset_key_calc]
        else:
            if last_pred_mu_calc.get(asset_key_calc) is not None:
                mu_ann = last_pred_mu_calc[asset_key_calc]
    holding_period_years = horizon_days_calc / TRADING_DAYS_PER_YEAR
    expected_h_period_return = (1 + mu_ann) ** holding_period_years - 1
    return expected_h_period_return


def optimize_portfolio_cvar(scenario_returns, expected_asset_returns_h,
                            cvar_epsilon_tail, target_portfolio_return_h):
    n_sim, num_assets = scenario_returns.shape
    c = np.zeros(num_assets + 1 + n_sim)
    c[num_assets] = 1
    c[num_assets + 1:] = 1 / (n_sim * cvar_epsilon_tail)
    A_ub_list = []
    b_ub_list = []
    for j in range(n_sim):
        row = np.zeros(num_assets + 1 + n_sim)
        row[:num_assets] = -scenario_returns[j, :]
        row[num_assets] = -1
        row[num_assets + 1 + j] = -1
        A_ub_list.append(row)
        b_ub_list.append(0)
    if target_portfolio_return_h is not None:
        exp_ret_row = np.zeros(num_assets + 1 + n_sim)
        exp_ret_row[:num_assets] = -expected_asset_returns_h
        A_ub_list.append(exp_ret_row)
        b_ub_list.append(-target_portfolio_return_h)
    A_ub = np.array(A_ub_list) if A_ub_list else None
    b_ub = np.array(b_ub_list) if b_ub_list else None
    A_eq = np.zeros((1, num_assets + 1 + n_sim))
    A_eq[0, :num_assets] = 1
    b_eq = np.array([1])
    bounds = [(0, None) for _ in range(num_assets)] + \
             [(None, None)] + \
             [(0, None) for _ in range(n_sim)]
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if result.success:
            weights = result.x[:num_assets]
            weights[np.abs(weights) < 1e-6] = 0
            if np.sum(weights) > 1e-6:
                weights = weights / np.sum(weights)
            else:
                return None
            return weights
        else:
            return None
    except Exception as e:
        return None


# --- 3. 执行滚动优化 与 风险度量计算 ---
print("\n--- 步骤 3: 执行滚动优化与组合/单资产风险度量计算 ---")
portfolio_target_h_period_return = (1 + PORTFOLIO_TARGET_ANNUAL_RETURN) ** (H_HORIZON_DAYS / TRADING_DAYS_PER_YEAR) - 1
print(f"投资组合持有期 {H_HORIZON_DAYS} 天的目标收益率: {portfolio_target_h_period_return:.4%}")

optimized_weights_list = []
individual_asset_risk_metrics_list = []
optimized_portfolio_risk_metrics_list = []
# 新增：用于存储每日组合预期收益与目标收益的列表
portfolio_expected_return_tracking_list = []

valid_simulation_dates = [d for d in SIMULATION_DATES if d in final_df_full.index and d in paths_by_date]

for t_start_date in tqdm(valid_simulation_dates, desc="滚动优化与风险计算进度"):
    s0_series = final_df_full.loc[t_start_date, [f'St_{key}' for key in asset_order_in_paths]]
    if s0_series.isnull().any():
        continue

    sim_paths_today = paths_by_date.get(t_start_date)
    if sim_paths_today is None:
        continue

    # A. 单资产风险度量计算
    for asset_idx, asset_k_risk in enumerate(asset_order_in_paths):
        s0_individual_asset = s0_series.iloc[asset_idx]
        individual_asset_paths = sim_paths_today[:, asset_idx, :]
        s_t_plus_h_individual = individual_asset_paths[-1, :]
        pnl_individual_scenarios = s_t_plus_h_individual - s0_individual_asset
        losses_individual_scenarios = -pnl_individual_scenarios
        if s0_individual_asset > 1e-9:
            var_individual_abs = np.percentile(losses_individual_scenarios, CVAR_ALPHA * 100)
            losses_beyond_var_individual = losses_individual_scenarios[
                losses_individual_scenarios >= var_individual_abs]
            cvar_individual_abs = losses_beyond_var_individual.mean() if losses_beyond_var_individual.size > 0 else var_individual_abs
            var_individual_pct = var_individual_abs / s0_individual_asset
            cvar_individual_pct = cvar_individual_abs / s0_individual_asset
            individual_asset_risk_metrics_list.append({
                'date': t_start_date, 'asset': asset_k_risk, 'S0': s0_individual_asset,
                'VaR_abs': var_individual_abs, 'CVaR_abs': cvar_individual_abs,
                'VaR_pct': var_individual_pct, 'CVaR_pct': cvar_individual_pct
            })

    # B. 投资组合优化部分
    s_t_plus_h_portfolio = sim_paths_today[-1, :, :]
    scenario_returns_np_portfolio = (s_t_plus_h_portfolio.T / s0_series.values) - 1

    expected_returns_h_period = np.zeros(len(asset_order_in_paths))
    for i, asset_k_opt in enumerate(asset_order_in_paths):  # 使用 asset_order_in_paths 保证顺序
        static_params_asset = all_base_params.get(asset_k_opt, {})
        ml_pipeline_asset = ml_models_pipelines.get(asset_k_opt)
        expected_returns_h_period[i] = calculate_expected_period_return(
            asset_k_opt, t_start_date, static_params_asset, ml_pipeline_asset,
            final_df_full, ml_feature_names, last_dynamic_mu_preds_for_E, H_HORIZON_DAYS
        )

    optimal_w = optimize_portfolio_cvar(scenario_returns_np_portfolio,
                                        expected_returns_h_period,
                                        CVAR_EPSILON_TAIL,
                                        portfolio_target_h_period_return)  # 这是持有期H的目标收益

    current_weights_dict = {'date': t_start_date}
    achieved_expected_portfolio_return_h_current = np.nan  # 初始化

    if optimal_w is not None:
        for i, asset_k_w in enumerate(asset_order_in_paths):  # 确保权重顺序与 asset_order_in_paths 一致
            current_weights_dict[f'w_{asset_k_w}'] = optimal_w[i]

        # 计算基于当日E[ri,H]和优化权重的组合预期收益
        achieved_expected_portfolio_return_h_current = np.sum(optimal_w * expected_returns_h_period)

        portfolio_scenario_returns = scenario_returns_np_portfolio @ optimal_w
        portfolio_scenario_losses = -portfolio_scenario_returns
        var_portfolio_optimized_h_period = np.percentile(portfolio_scenario_losses, CVAR_ALPHA * 100)
        losses_beyond_var_portfolio = portfolio_scenario_losses[
            portfolio_scenario_losses >= var_portfolio_optimized_h_period]
        cvar_portfolio_optimized_h_period = losses_beyond_var_portfolio.mean() if losses_beyond_var_portfolio.size > 0 else var_portfolio_optimized_h_period
        optimized_portfolio_risk_metrics_list.append({
            'date': t_start_date,
            'VaR_portfolio_H_return': var_portfolio_optimized_h_period,
            'CVaR_portfolio_H_return': cvar_portfolio_optimized_h_period
        })
    else:
        default_weight = 1.0 / len(asset_order_in_paths)
        for asset_k_w in asset_order_in_paths:
            current_weights_dict[f'w_{asset_k_w}'] = default_weight
        # 优化失败时，achieved_expected_portfolio_return_h_current 可以基于等权重计算
        achieved_expected_portfolio_return_h_current = np.sum(
            np.array([default_weight] * len(asset_order_in_paths)) * expected_returns_h_period)
        optimized_portfolio_risk_metrics_list.append({
            'date': t_start_date, 'VaR_portfolio_H_return': np.nan, 'CVaR_portfolio_H_return': np.nan
        })

    optimized_weights_list.append(current_weights_dict)
    portfolio_expected_return_tracking_list.append({
        'date': t_start_date,
        'achieved_expected_H_return': achieved_expected_portfolio_return_h_current,
        'target_H_return': portfolio_target_h_period_return  # 这是固定的持有期目标
    })

# --- 4. 保存结果与绘图 ---
print("\n--- 步骤 4: 保存结果与绘图 ---")

# 4.1 保存优化权重并绘制堆叠面积图
optimized_weights_df = pd.DataFrame(optimized_weights_list)
if not optimized_weights_df.empty:
    optimized_weights_df.set_index('date', inplace=True)
    optimized_weights_df.to_csv("optimized_portfolio_weights_cvar.csv")
    print("优化后的投资组合权重已保存到 optimized_portfolio_weights_cvar.csv")
    print("优化权重预览:")
    print(optimized_weights_df.head())

    print("\n正在绘制优化后投资组合权重的时间序列图...")
    plot_weight_cols = [WEIGHT_COL_NAMES[key] for key in MAIN_ASSETS_KEYS if
                        WEIGHT_COL_NAMES[key] in optimized_weights_df.columns]
    if plot_weight_cols:
        plt.figure(figsize=(14, 7))
        optimized_weights_df[plot_weight_cols].plot.area(ax=plt.gca(), alpha=0.7)
        plt.title('Optimized Portfolio Weights Over Time (CVaR Strategy)')
        plt.ylabel('Portfolio Weight');
        plt.xlabel('Date');
        plt.ylim(0, 1)
        # 使用 MAIN_ASSETS_KEYS 生成图例标签，确保与 plot_weight_cols 顺序对应
        legend_labels = [key.upper() for key in MAIN_ASSETS_KEYS if WEIGHT_COL_NAMES[key] in plot_weight_cols]
        plt.legend(title='Assets', labels=legend_labels)
        plt.grid(True, linestyle='--', alpha=0.7);
        plt.tight_layout()
        plt.savefig("optimized_portfolio_weights_timeseries.png")
        print("图表已保存: optimized_portfolio_weights_timeseries.png")
        plt.show()
    else:
        print("警告: optimized_weights_df 中缺少用于绘图的权重列。")
else:
    print("未能生成任何优化权重。")

# 新增：4.1b 绘制平均权重饼图
if not optimized_weights_df.empty and plot_weight_cols:  # 确保有权重数据和列名
    average_weights = optimized_weights_df[plot_weight_cols].mean()
    plt.figure(figsize=(8, 8))
    plt.pie(average_weights, labels=[col.replace('w_', '').upper() for col in average_weights.index], autopct='%1.1f%%',
            startangle=90, pctdistance=0.85)
    plt.title('Average Portfolio Allocation Over Backtest Period')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig("average_portfolio_allocation_pie.png")
    print("图表已保存: average_portfolio_allocation_pie.png")
    plt.show()
else:
    print("无法计算或绘制平均权重饼图，因权重数据为空或列名不匹配。")

# 4.2 保存并绘制单资产风险度量
# (与上一版本相同，此处省略以节省空间，确保绘图逻辑正确处理子图)
individual_asset_risk_metrics_df = pd.DataFrame(individual_asset_risk_metrics_list)
if not individual_asset_risk_metrics_df.empty:
    individual_asset_risk_metrics_df.set_index('date', inplace=True)
    individual_asset_risk_metrics_df.to_csv("individual_asset_risk_metrics.csv")
    print("\n单资产风险度量已保存到 individual_asset_risk_metrics.csv")
    print("单资产风险度量预览 (平均值):")
    print(individual_asset_risk_metrics_df.groupby('asset')[['VaR_pct', 'CVaR_pct']].mean())
    print("\n正在绘制单资产VaR和CVaR时间序列图...")
    unique_assets_for_plot = individual_asset_risk_metrics_df['asset'].unique()
    num_assets_plot = len(unique_assets_for_plot)
    if num_assets_plot > 0:
        fig_h = 5 * num_assets_plot
        fig, axes = plt.subplots(num_assets_plot, 2, figsize=(15, fig_h), sharex=True)
        if num_assets_plot == 1: axes = np.array([axes])  # 确保axes总是二维的
        axes_flat = axes.flatten()
        fig.suptitle(f'{H_HORIZON_DAYS}-Day Individual Asset Risk Metrics ({CVAR_ALPHA * 100:.0f}%)', fontsize=16)
        plot_idx = 0
        for asset_k_plot in unique_assets_for_plot:  # 应该迭代 actual_asset_order 或 MAIN_ASSETS_KEYS
            asset_data_for_plot = individual_asset_risk_metrics_df[
                individual_asset_risk_metrics_df['asset'] == asset_k_plot]
            if not asset_data_for_plot.empty:
                ax_var = axes_flat[plot_idx]
                ax_var.plot(asset_data_for_plot.index, asset_data_for_plot['VaR_pct'],
                            label=f'{asset_k_plot.upper()} VaR % S0', color='blue', alpha=0.7)
                ax_var.set_ylabel(f'{asset_k_plot.upper()} VaR (% S0)');
                ax_var.legend(loc='upper left');
                ax_var.grid(True);
                ax_var.tick_params(axis='x', rotation=30, labelsize=8)
                plot_idx += 1
                ax_cvar = axes_flat[plot_idx]
                ax_cvar.plot(asset_data_for_plot.index, asset_data_for_plot['CVaR_pct'],
                             label=f'{asset_k_plot.upper()} CVaR % S0', color='red', alpha=0.7)
                ax_cvar.set_ylabel(f'{asset_k_plot.upper()} CVaR (% S0)');
                ax_cvar.legend(loc='upper left');
                ax_cvar.grid(True);
                ax_cvar.tick_params(axis='x', rotation=30, labelsize=8)
                plot_idx += 1
            else:
                if plot_idx < len(axes_flat): axes_flat[plot_idx].set_title(
                    f"{asset_k_plot.upper()} No VaR Data"); plot_idx += 1
                if plot_idx < len(axes_flat): axes_flat[plot_idx].set_title(
                    f"{asset_k_plot.upper()} No CVaR Data"); plot_idx += 1
        fig.autofmt_xdate();
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]);
        plt.savefig("individual_asset_risk_timeseries.png");
        plt.show()
else:
    print("未能计算任何单资产风险度量。")

# 4.3 保存并绘制优化后投资组合的VaR/CVaR时间序列图
# (与上一版本相同，此处省略以节省空间)
if optimized_portfolio_risk_metrics_list:
    optimized_portfolio_risk_df = pd.DataFrame(optimized_portfolio_risk_metrics_list)
    if not optimized_portfolio_risk_df.empty:
        optimized_portfolio_risk_df.set_index('date', inplace=True)
        optimized_portfolio_risk_df.dropna(inplace=True)
        if not optimized_portfolio_risk_df.empty:
            optimized_portfolio_risk_df.to_csv("optimized_portfolio_risk_metrics.csv")
            print("\n优化后投资组合的VaR/CVaR (基于H期收益率)已保存到 optimized_portfolio_risk_metrics.csv")
            print("优化后投资组合风险预览 (均值):");
            print(optimized_portfolio_risk_df.mean())
            plt.figure(figsize=(15, 7))
            plt.plot(optimized_portfolio_risk_df.index, optimized_portfolio_risk_df['VaR_portfolio_H_return'],
                     label=f'Optimized Portfolio {CVAR_ALPHA * 100:.0f}% VaR (H-period Return)', color='purple',
                     alpha=0.8)
            plt.plot(optimized_portfolio_risk_df.index, optimized_portfolio_risk_df['CVaR_portfolio_H_return'],
                     label=f'Optimized Portfolio {CVAR_ALPHA * 100:.0f}% CVaR (H-period Return)', color='green',
                     alpha=0.8, linestyle='--')
            plt.title(f'{H_HORIZON_DAYS}-Day Optimized Portfolio Risk (Based on H-period Returns)');
            plt.ylabel('Risk Value (Return Scale)');
            plt.xlabel('Date')
            plt.legend();
            plt.grid(True);
            plt.gca().tick_params(axis='x', rotation=30, labelsize=8);
            # fig.autofmt_xdate() # fig is not defined here, use plt.gcf() if needed or apply to specific axes
            plt.gcf().autofmt_xdate()
            plt.tight_layout();
            plt.savefig("optimized_portfolio_risk_timeseries.png");
            plt.show()
        else:
            print("所有优化组合风险度量均为NaN，无法绘图或保存。")
    else:
        print("未能计算任何优化后投资组合的风险度量。")

# 新增：4.4 绘制每日优化组合的预期收益与目标收益对比图
if portfolio_expected_return_tracking_list:
    portfolio_return_tracking_df = pd.DataFrame(portfolio_expected_return_tracking_list)
    if not portfolio_return_tracking_df.empty:
        portfolio_return_tracking_df.set_index('date', inplace=True)
        portfolio_return_tracking_df.to_csv("portfolio_expected_vs_target_return.csv")
        print("\n每日组合预期收益与目标收益对比数据已保存到 portfolio_expected_vs_target_return.csv")

        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_return_tracking_df.index, portfolio_return_tracking_df['achieved_expected_H_return'],
                 label='Achieved Portfolio E[R_H] (from model inputs)', color='dodgerblue', alpha=0.8)
        plt.plot(portfolio_return_tracking_df.index, portfolio_return_tracking_df['target_H_return'],
                 label=f'Target Portfolio E[R_H] ({portfolio_target_h_period_return:.4%})', color='red', linestyle='--',
                 alpha=0.7)
        plt.title(f'Optimized Portfolio: Achieved vs. Target Expected {H_HORIZON_DAYS}-Day Return')
        plt.ylabel(f'{H_HORIZON_DAYS}-Day Expected Return')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.gca().tick_params(axis='x', rotation=30, labelsize=8)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig("portfolio_achieved_vs_target_return.png")
        print("图表已保存: portfolio_achieved_vs_target_return.png")
        plt.show()
    else:
        print("未能生成组合预期收益与目标收益的对比数据。")
else:
    print("未能收集到组合预期收益与目标收益的跟踪数据。")

print("\n--- 阶段五 (CVaR投资组合优化与组合/单资产风险度量) 完成 ---")