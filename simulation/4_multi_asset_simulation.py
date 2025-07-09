import pandas as pd
import numpy as np
import joblib
import json
from tqdm import tqdm
import matplotlib.pyplot as plt  # 新增导入
import seaborn as sns  # 新增导入

# --- 0. 定义常量与加载前期结果 ---
print("--- 阶段四：多资产价格路径联合仿真 ---")

MAIN_ASSETS_KEYS = ['etf', 'gbi', 'cea']
EXPECTED_ML_FEATURE_COLS = ['lag1_r_j', 'lag3_r_j', 'lag5_r_j', 'roll_mean5_r_j',
                            'lag1_r_fu', 'lag3_r_fu', 'lag5_r_fu', 'roll_mean5_r_fu',
                            'lag1_r_jm', 'lag3_r_jm', 'lag5_r_jm', 'roll_mean5_r_jm']
RETURN_COL_NAMES = {
    'etf': 'r_etf',
    'gbi': 'r_gbi',
    'cea': 'r_cea'
}
N_SIMULATIONS = 10000
H_HORIZON_DAYS = 20
TRADING_DAYS_PER_YEAR = 252
DT = 1 / TRADING_DAYS_PER_YEAR
CORRELATION_WINDOW = 60

# 加载静态参数
try:
    with open('all_assets_base_params.json', 'r') as f:
        all_base_params = json.load(f)
    print("所有资产的基础静态参数 (all_assets_base_params.json) 加载成功。")
except FileNotFoundError:
    print("错误: all_assets_base_params.json 文件未找到。请先运行阶段二。")
    exit()

# 加载ML模型 (ETF 和 CEA)
ml_models = {}
ml_feature_names = None
for asset_key in ['etf', 'cea']:
    model_filename = f'dynamic_mu_{asset_key}_model.joblib'
    try:
        model_artifacts = joblib.load(model_filename)
        ml_models[asset_key] = model_artifacts['pipeline']
        if ml_feature_names is None:
            ml_feature_names = model_artifacts.get('feature_names', EXPECTED_ML_FEATURE_COLS)
        print(f"资产 '{asset_key.upper()}' 的ML模型 ({model_filename}) 加载成功。")
    except FileNotFoundError:
        print(f"警告: 资产 '{asset_key.upper()}' 的ML模型 {model_filename} 未找到。该资产将仅使用静态mu。")
        ml_models[asset_key] = None
    except Exception as e:
        print(f"加载 {model_filename} 时发生错误: {e}")
        ml_models[asset_key] = None

if ml_feature_names is None:
    print("警告: 未能从任何ML模型文件中加载特征名列表，将使用预定义的特征列表。")
    ml_feature_names = EXPECTED_ML_FEATURE_COLS

# 加载完整数据集
try:
    final_df_full = pd.read_csv("final_data_for_modeling_multi_asset.csv", index_col='date', parse_dates=True)
    missing_features = [col for col in ml_feature_names if col not in final_df_full.columns]
    if missing_features:
        print(f"错误: final_df_full 中缺少以下必要的ML特征列: {missing_features}。")
        # exit() # 实际使用时，如果这里报错，应该确保阶段三正确保存了特征
    print("final_data_for_modeling_multi_asset.csv 加载成功。")
except FileNotFoundError:
    print("错误: final_data_for_modeling_multi_asset.csv 文件未找到。请先运行阶段一。")
    exit()

# 加载阶段C和D数据
try:
    data_C = pd.read_csv("data_C_multi_asset_dynamic_eval.csv", index_col='date', parse_dates=True)
    print(f"阶段 C 数据加载成功，共 {len(data_C)} 条。")
except FileNotFoundError:
    data_C = pd.DataFrame()
    print("警告: data_C_multi_asset_dynamic_eval.csv 文件未找到。阶段C的仿真将受影响。")

try:
    data_D = pd.read_csv("data_D_multi_asset_static_ext_eval.csv", index_col='date', parse_dates=True)
    print(f"阶段 D 数据加载成功，共 {len(data_D)} 条。")
except FileNotFoundError:
    data_D = pd.DataFrame()
    print("警告: data_D_multi_asset_static_ext_eval.csv 文件未找到。阶段D的仿真将受影响。")

ACTUAL_FUTURES_END_DATE_ADJ_STR = all_base_params.get('ACTUAL_FUTURES_END_DATE_ADJ_STR', '2024-09-05')
try:
    LAST_REAL_FUTURES_DATA_DATE_FOR_MU = pd.to_datetime(ACTUAL_FUTURES_END_DATE_ADJ_STR)
    print(
        f"动态模型mu_dyn参数固定的参考日期 (能源期货最后有效日 for ML input): {LAST_REAL_FUTURES_DATA_DATE_FOR_MU.strftime('%Y-%m-%d')}")
except ValueError:
    print(f"错误: 'ACTUAL_FUTURES_END_DATE_ADJ_STR' 日期格式不正确: {ACTUAL_FUTURES_END_DATE_ADJ_STR}。将使用默认值。")
    LAST_REAL_FUTURES_DATA_DATE_FOR_MU = pd.to_datetime('2024-09-05')

last_dynamic_mu_preds = {key: None for key in MAIN_ASSETS_KEYS}


# --- 1. 动态参数预测函数 ---
def predict_dynamic_mu_for_asset_simulation(current_date_for_features, asset_key_local,
                                            ml_model_pipeline, static_mu_val,
                                            all_features_df, trained_feature_names_list):
    if ml_model_pipeline is None: return static_mu_val
    if current_date_for_features not in all_features_df.index: return static_mu_val
    features_row = all_features_df.loc[[current_date_for_features], trained_feature_names_list]
    if features_row.empty or features_row.isnull().values.any().any(): return static_mu_val
    try:
        return ml_model_pipeline.predict(features_row)[0]
    except Exception:
        return static_mu_val


# --- 2. 多资产蒙特卡洛仿真函数 ---
def multi_asset_path_simulation(s0_vector, params_dict, corr_cholesky_L,
                                dt_sim, horizon_steps, n_sims):
    num_assets = len(s0_vector)
    paths = np.zeros((horizon_steps + 1, num_assets, n_sims))
    paths[0, :, :] = s0_vector[:, np.newaxis]
    for t_step in range(1, horizon_steps + 1):
        indep_norm_rv = np.random.standard_normal((num_assets, n_sims))
        corr_norm_rv = corr_cholesky_L @ indep_norm_rv
        for asset_idx, asset_key_sim in enumerate(MAIN_ASSETS_KEYS):
            asset_params = params_dict[asset_key_sim]
            current_s = paths[t_step - 1, asset_idx, :]
            mu_sim, sigma_sim = asset_params['mu_sim'], asset_params['sigma_base']
            if asset_params['model_type'] == 'gbm':
                drift = (mu_sim - 0.5 * sigma_sim ** 2) * dt_sim
                diffusion = sigma_sim * np.sqrt(dt_sim) * corr_norm_rv[asset_idx, :]
                paths[t_step, asset_idx, :] = current_s * np.exp(drift + diffusion)
            elif asset_params['model_type'] == 'merton':
                lambda_sim, mj_sim, sigmaj_sim, k_avg_sim = asset_params['lambda_base'], asset_params['mJ_base'], \
                asset_params['sigmaJ_base'], asset_params['k_avg_base']
                drift = (mu_sim - 0.5 * sigma_sim ** 2 - lambda_sim * k_avg_sim) * dt_sim
                diffusion = sigma_sim * np.sqrt(dt_sim) * corr_norm_rv[asset_idx, :]
                poisson_draws = np.random.poisson(lambda_sim * dt_sim, n_sims)
                jump_comp = np.zeros(n_sims)
                for k_sim_idx in range(n_sims):
                    if poisson_draws[k_sim_idx] > 0:
                        log_jumps_this_step = np.random.normal(mj_sim, sigmaj_sim, poisson_draws[k_sim_idx])
                        jump_comp[k_sim_idx] = np.sum(log_jumps_this_step)
                paths[t_step, asset_idx, :] = current_s * np.exp(drift + diffusion + jump_comp)
    return paths


# --- 3. 执行滚动仿真 ---
print("\n--- 步骤 3: 执行滚动仿真 ---")
if not data_C.empty and not data_D.empty:
    evaluation_dates = data_C.index.union(data_D.index).sort_values()
elif not data_C.empty:
    evaluation_dates = data_C.index.sort_values()
elif not data_D.empty:
    evaluation_dates = data_D.index.sort_values()
else:
    print("错误: 阶段C和阶段D数据均为空，无法进行仿真。")
    exit()
print(f"将对 {len(evaluation_dates)} 个日期进行滚动仿真。")

all_simulated_paths_by_date = {}
asset_return_cols_for_corr = [RETURN_COL_NAMES[key] for key in MAIN_ASSETS_KEYS]
# 新增：用于存储每日使用的相关系数
daily_used_correlations_list = []

for t_start_date in tqdm(evaluation_dates, desc="滚动仿真进度"):
    if t_start_date not in final_df_full.index: continue
    s0_values = []
    valid_s0 = True
    for asset_key_s0 in MAIN_ASSETS_KEYS:
        st_col_s0 = f'St_{asset_key_s0}'
        if st_col_s0 in final_df_full.columns:
            s0_val = final_df_full.loc[t_start_date, st_col_s0]
            if pd.isna(s0_val): valid_s0 = False; break
            s0_values.append(s0_val)
        else:
            valid_s0 = False; break
    if not valid_s0 or len(s0_values) != len(MAIN_ASSETS_KEYS): continue
    s0_vector_np = np.array(s0_values)

    corr_matrix_current = np.identity(len(MAIN_ASSETS_KEYS))
    idx_loc_t_start = final_df_full.index.get_loc(t_start_date)
    if idx_loc_t_start >= CORRELATION_WINDOW:
        hist_returns_for_corr = final_df_full.iloc[idx_loc_t_start - CORRELATION_WINDOW: idx_loc_t_start][
            asset_return_cols_for_corr]
        hist_returns_for_corr.dropna(inplace=True)
        if len(hist_returns_for_corr) >= 10 and not hist_returns_for_corr.empty:
            if (hist_returns_for_corr.std() > 1e-9).all():
                temp_corr = hist_returns_for_corr.corr()
                if not temp_corr.isnull().values.any():
                    corr_matrix_current = temp_corr.values

    # 存储当期使用的相关系数
    daily_used_correlations_list.append({
        'date': t_start_date,
        'corr_etf_gbi': corr_matrix_current[MAIN_ASSETS_KEYS.index('etf'), MAIN_ASSETS_KEYS.index('gbi')],
        'corr_etf_cea': corr_matrix_current[MAIN_ASSETS_KEYS.index('etf'), MAIN_ASSETS_KEYS.index('cea')],
        'corr_gbi_cea': corr_matrix_current[MAIN_ASSETS_KEYS.index('gbi'), MAIN_ASSETS_KEYS.index('cea')]
    })

    try:
        cholesky_L_current = np.linalg.cholesky(corr_matrix_current)
    except np.linalg.LinAlgError:
        cholesky_L_current = np.identity(len(MAIN_ASSETS_KEYS))

    simulation_params_for_assets = {}
    for asset_key_param in MAIN_ASSETS_KEYS:
        asset_static_params = all_base_params.get(asset_key_param)
        if not asset_static_params: break
        current_mu_for_sim = asset_static_params['mu_base']
        if asset_key_param in ml_models and ml_models[asset_key_param] is not None:
            if t_start_date <= LAST_REAL_FUTURES_DATA_DATE_FOR_MU:
                predicted_mu = predict_dynamic_mu_for_asset_simulation(
                    t_start_date, asset_key_param, ml_models[asset_key_param],
                    asset_static_params['mu_base'], final_df_full, ml_feature_names
                )
                if predicted_mu is not None and not pd.isna(predicted_mu):
                    current_mu_for_sim = predicted_mu
                    last_dynamic_mu_preds[asset_key_param] = current_mu_for_sim
                elif last_dynamic_mu_preds.get(asset_key_param) is not None:
                    current_mu_for_sim = last_dynamic_mu_preds[asset_key_param]
            else:
                current_mu_for_sim = last_dynamic_mu_preds.get(asset_key_param, asset_static_params['mu_base'])
                if current_mu_for_sim is None: current_mu_for_sim = asset_static_params['mu_base']
        simulation_params_for_assets[asset_key_param] = {**asset_static_params, 'mu_sim': current_mu_for_sim}
    else:
        sim_paths_all_assets_today = multi_asset_path_simulation(
            s0_vector_np, simulation_params_for_assets, cholesky_L_current,
            DT, H_HORIZON_DAYS, N_SIMULATIONS
        )
        all_simulated_paths_by_date[t_start_date.strftime('%Y-%m-%d')] = sim_paths_all_assets_today

print(f"\n滚动仿真完成。共为 {len(all_simulated_paths_by_date)} 个日期生成了联合仿真路径。")

# --- 4. 保存仿真结果 ---
simulation_results_to_save = {
    'paths_by_date': all_simulated_paths_by_date,
    'asset_order': MAIN_ASSETS_KEYS,
    'horizon_days': H_HORIZON_DAYS,
    'n_simulations': N_SIMULATIONS,
    'simulation_dates': list(all_simulated_paths_by_date.keys())
}
try:
    joblib.dump(simulation_results_to_save, 'simulation_results_multi_asset.joblib')
    print("\n多资产仿真结果已保存到 simulation_results_multi_asset.joblib")
except Exception as e:
    print(f"保存仿真结果时发生错误: {e}")

# --- 5. 新增可视化 ---
print("\n--- 步骤 5: 生成额外的可视化图表 ---")

# 5.1 滚动相关性实现的可视化
if daily_used_correlations_list:
    rolling_corr_df = pd.DataFrame(daily_used_correlations_list)
    rolling_corr_df.set_index('date', inplace=True)

    plt.figure(figsize=(14, 7))
    for col in ['corr_etf_gbi', 'corr_etf_cea', 'corr_gbi_cea']:
        if col in rolling_corr_df.columns:
            plt.plot(rolling_corr_df.index, rolling_corr_df[col], label=col.replace('corr_', '').upper(), alpha=0.7)

    plt.title(f'{CORRELATION_WINDOW}-Day Rolling Correlation Coefficients Used in Simulation')
    plt.xlabel('Date (Start of Simulation Horizon)')
    plt.ylabel('Correlation Coefficient')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("simulation_rolling_correlations_used.png")
    print("图表已保存: simulation_rolling_correlations_used.png")
    plt.show()
else:
    print("未能收集到足够的滚动相关性数据进行绘图。")

# 5.2 样本路径图和期末价格分布图 (选择一个示例日期)
if all_simulated_paths_by_date:
    example_date_str = next(iter(all_simulated_paths_by_date))  # 取第一个有仿真结果的日期作为示例
    example_paths = all_simulated_paths_by_date[example_date_str]  # (H+1, num_assets, N_sims)
    example_s0 = []
    for asset_key_ex_s0 in MAIN_ASSETS_KEYS:  # 获取示例日期的S0
        st_col_ex_s0 = f'St_{asset_key_ex_s0}'
        if pd.to_datetime(example_date_str) in final_df_full.index and st_col_ex_s0 in final_df_full.columns:
            example_s0.append(final_df_full.loc[pd.to_datetime(example_date_str), st_col_ex_s0])
        else:
            example_s0.append(np.nan)  # 如果找不到S0，则标记

    num_sample_paths_to_plot = 5
    path_indices_to_plot = np.random.choice(N_SIMULATIONS, num_sample_paths_to_plot, replace=False)

    # 样本路径图
    fig_sample, axes_sample = plt.subplots(len(MAIN_ASSETS_KEYS), 1, figsize=(14, 4 * len(MAIN_ASSETS_KEYS)),
                                           sharex=True)
    if len(MAIN_ASSETS_KEYS) == 1: axes_sample = [axes_sample]  # 确保可迭代
    fig_sample.suptitle(f'Sample Simulated Price Paths (Starting {example_date_str}, {num_sample_paths_to_plot} paths)',
                        fontsize=16)
    time_steps = np.arange(H_HORIZON_DAYS + 1)

    for i, asset_key_plot in enumerate(MAIN_ASSETS_KEYS):
        asset_paths_sample = example_paths[:, i, path_indices_to_plot]  # (H+1, num_sample_paths_to_plot)
        axes_sample[i].plot(time_steps, asset_paths_sample, alpha=0.7)
        axes_sample[i].set_title(f'{asset_key_plot.upper()} - S0: {example_s0[i]:.2f}' if not pd.isna(
            example_s0[i]) else f'{asset_key_plot.upper()}')
        axes_sample[i].set_ylabel('Simulated Price')
        axes_sample[i].grid(True)
    axes_sample[-1].set_xlabel('Time Steps (Days into Horizon)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"sample_simulated_paths_{example_date_str.replace('-', '')}.png")
    print(f"图表已保存: sample_simulated_paths_{example_date_str.replace('-', '')}.png")
    plt.show()

    # 期末价格分布图
    fig_dist, axes_dist = plt.subplots(len(MAIN_ASSETS_KEYS), 1, figsize=(10, 4 * len(MAIN_ASSETS_KEYS)), sharey=False)
    if len(MAIN_ASSETS_KEYS) == 1: axes_dist = [axes_dist]  # 确保可迭代
    fig_dist.suptitle(
        f'Distribution of Simulated Terminal Prices (Starting {example_date_str}, Horizon: {H_HORIZON_DAYS} days)',
        fontsize=16)

    for i, asset_key_plot in enumerate(MAIN_ASSETS_KEYS):
        terminal_prices_asset = example_paths[-1, i, :]  # (N_sims,)
        sns.histplot(terminal_prices_asset, kde=True, ax=axes_dist[i], bins=50, stat="density")
        axes_dist[i].set_title(f'{asset_key_plot.upper()} - S0: {example_s0[i]:.2f}' if not pd.isna(
            example_s0[i]) else f'{asset_key_plot.upper()}')
        axes_dist[i].set_xlabel('Simulated Terminal Price')
        axes_dist[i].set_ylabel('Density')
        axes_dist[i].grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"terminal_price_distribution_{example_date_str.replace('-', '')}.png")
    print(f"图表已保存: terminal_price_distribution_{example_date_str.replace('-', '')}.png")
    plt.show()
else:
    print("没有可用的仿真路径来生成示例图表。")

print("\n--- 阶段四 (多资产情景生成 与 额外可视化) 完成 ---")