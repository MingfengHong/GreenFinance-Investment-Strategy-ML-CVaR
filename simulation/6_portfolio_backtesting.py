import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 0. 定义常量与配置 ---
print("--- 阶段六：投资组合回溯检验与绩效评估 ---")

MAIN_ASSETS_KEYS = ['etf', 'gbi', 'cea']
RETURN_COL_NAMES = {
    'etf': 'r_etf',
    'gbi': 'r_gbi',
    'cea': 'r_cea'
}
WEIGHT_COL_NAMES = {
    'etf': 'w_etf',
    'gbi': 'w_gbi',
    'cea': 'w_cea'
}

TRADING_DAYS_PER_YEAR = 252
ANNUAL_RISK_FREE_RATE = 0.015
# DAILY_RISK_FREE_RATE (简单日无风险利率) 用于计算简单超额收益
DAILY_RISK_FREE_RATE = (1 + ANNUAL_RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1
# DAILY_LOG_RISK_FREE_RATE (对数日无风险利率) 如果需要与对数收益率直接做差
# DAILY_LOG_RISK_FREE_RATE = np.log(1 + DAILY_RISK_FREE_RATE)


# --- 1. 加载所需数据 ---
print("\n--- 步骤 1: 加载数据 ---")
try:
    optimized_weights_df = pd.read_csv("optimized_portfolio_weights_cvar.csv", index_col='date', parse_dates=True)
    print("优化后的投资组合权重 (optimized_portfolio_weights_cvar.csv) 加载成功。")
except FileNotFoundError:
    print("错误: optimized_portfolio_weights_cvar.csv 文件未找到。请先运行阶段五。")
    exit()

try:
    final_df_full = pd.read_csv("final_data_for_modeling_multi_asset.csv", index_col='date', parse_dates=True)
    # actual_returns_df 存储的是对数收益率 (log returns)
    actual_log_returns_df = final_df_full[[RETURN_COL_NAMES[key] for key in MAIN_ASSETS_KEYS]].copy()
    actual_log_returns_df.columns = MAIN_ASSETS_KEYS
    print("实际资产日对数收益率数据加载并准备成功。")
except FileNotFoundError:
    print("错误: final_data_for_modeling_multi_asset.csv 文件未找到。")
    exit()
except KeyError as e:
    print(f"错误: final_data_for_modeling_multi_asset.csv 中缺少必要的收益率列: {e}")
    exit()

common_dates = optimized_weights_df.index.intersection(actual_log_returns_df.index)
if common_dates.empty:
    print("错误: 优化权重和实际收益率数据之间没有共同的日期，无法进行回测。")
    exit()

weights_for_backtest = optimized_weights_df.loc[common_dates]
# actual_log_returns_for_backtest 是 T+1 日的对数收益率，其索引对齐到 T 日
actual_log_returns_shifted = actual_log_returns_df.shift(-1)
actual_log_returns_for_backtest = actual_log_returns_shifted.loc[weights_for_backtest.index]

valid_backtest_mask = actual_log_returns_for_backtest.notna().all(axis=1)
weights_for_backtest = weights_for_backtest[valid_backtest_mask]
actual_log_returns_for_backtest = actual_log_returns_for_backtest[valid_backtest_mask]

if weights_for_backtest.empty or actual_log_returns_for_backtest.empty:
    print("错误: 对齐权重和收益率后，没有有效的回测数据。")
    exit()

print(
    f"回测将在 {weights_for_backtest.index.min().strftime('%Y-%m-%d')} 到 {weights_for_backtest.index.max().strftime('%Y-%m-%d')} 期间进行。")
print(f"有效回测期交易日数: {len(weights_for_backtest)}")

# --- 2. 计算各策略的日对数收益率 ---
print("\n--- 步骤 2: 计算各策略的日对数收益率 ---")
# portfolio_daily_log_returns DataFrame 将存储每个策略的日对数收益率
portfolio_daily_log_returns = pd.DataFrame(index=weights_for_backtest.index)

cvar_strategy_weights = weights_for_backtest[[WEIGHT_COL_NAMES[key] for key in MAIN_ASSETS_KEYS]]
cvar_strategy_weights.columns = MAIN_ASSETS_KEYS
# T日权重 * T+1日资产对数收益率 = T+1日组合对数收益率 (这是一个近似，严格来说对数收益率不能简单加权平均)
# 更准确的做法是：先计算T+1日各资产的价值因子 (1+简单收益) 或 exp(对数收益)
# 然后计算组合的价值因子，再取对数。
# 但如果日对数收益率很小，直接加权平均是常用的近似。我们这里先沿用这个近似。
# 如果要非常精确，应该用简单收益率计算组合日收益，然后再转回对数或直接用简单收益率计算后续指标。
# 为了与您的审计对齐，我们假设 portfolio_daily_returns 计算得到的是组合的“等效”日对数收益率
portfolio_daily_log_returns['CVaR_Optimized'] = (
            cvar_strategy_weights * actual_log_returns_for_backtest[MAIN_ASSETS_KEYS]).sum(axis=1)

num_assets = len(MAIN_ASSETS_KEYS)
equal_weights = np.array([1 / num_assets] * num_assets)
portfolio_daily_log_returns['Equal_Weight'] = (actual_log_returns_for_backtest[MAIN_ASSETS_KEYS] * equal_weights).sum(
    axis=1)

for asset_key in MAIN_ASSETS_KEYS:
    portfolio_daily_log_returns[f'BuyHold_{asset_key.upper()}'] = actual_log_returns_for_backtest[asset_key]

print("各策略日对数收益率计算完成。")
print(portfolio_daily_log_returns.head())

# --- 3. 计算绩效指标 (基于对数收益率进行修正) ---
print("\n--- 步骤 3: 计算绩效指标 ---")
performance_metrics = {}
drawdown_series_all_strategies = {}
nav_series_all_strategies = {}  # 新增：存储每个策略的净值序列

for strategy_name in portfolio_daily_log_returns.columns:
    daily_log_returns_strat = portfolio_daily_log_returns[strategy_name].copy().fillna(0)

    # 3.1 累计净值 (NAV) (基于对数收益率)
    # 假设初始投资为1，净值曲线 = exp(累计对数收益)
    nav_strat = np.exp(daily_log_returns_strat.cumsum())
    nav_series_all_strategies[strategy_name] = nav_strat  # 存储净值序列

    # 3.2 总回报率 (基于最终净值)
    final_net_value_strat = nav_strat.iloc[-1]
    total_return_strat = final_net_value_strat - 1

    # 3.3 年化回报率 (基于最终净值和年数)
    num_years_strat = len(daily_log_returns_strat) / TRADING_DAYS_PER_YEAR
    annualized_return_strat = (final_net_value_strat) ** (1 / num_years_strat) - 1 if num_years_strat > 0 else 0

    # 3.4 年化波动率 (基于日对数收益率的标准差，这是标准做法)
    annualized_log_volatility_strat = daily_log_returns_strat.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # 3.5 夏普比率 (基于日简单算术收益率计算，更标准)
    daily_simple_returns_strat = np.exp(daily_log_returns_strat) - 1
    excess_simple_returns_strat = daily_simple_returns_strat - DAILY_RISK_FREE_RATE
    # 年化简单算术收益率的波动率
    annualized_simple_volatility_strat = daily_simple_returns_strat.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    if annualized_simple_volatility_strat > 1e-9:  # 避免除以0
        sharpe_ratio_strat = (
                                         excess_simple_returns_strat.mean() * TRADING_DAYS_PER_YEAR) / annualized_simple_volatility_strat
    else:
        sharpe_ratio_strat = 0

    # 3.6 最大回撤 (基于正确计算的NAV)
    cumulative_max_nav_strat = nav_strat.cummax()
    drawdown_strat_series = (nav_strat - cumulative_max_nav_strat) / cumulative_max_nav_strat
    max_drawdown_strat = drawdown_strat_series.min()

    drawdown_series_all_strategies[strategy_name] = drawdown_strat_series

    performance_metrics[strategy_name] = {
        'Total Return': total_return_strat,
        'Annualized Return': annualized_return_strat,
        'Annualized Log Volatility': annualized_log_volatility_strat,  # 明确是基于对数收益率的波动率
        'Annualized Simple Volatility': annualized_simple_volatility_strat,  # 新增简单收益率波动
        'Sharpe Ratio (Simple Returns)': sharpe_ratio_strat,  # 明确是基于简单收益率
        'Max Drawdown': max_drawdown_strat,
        'Final Net Value': final_net_value_strat
    }

performance_metrics_df = pd.DataFrame(performance_metrics).T
print("\n各策略绩效指标:")
# 为了更清晰地显示，调整打印的列顺序和格式
display_cols = ['Final Net Value', 'Total Return', 'Annualized Return',
                'Annualized Log Volatility', 'Annualized Simple Volatility',
                'Sharpe Ratio (Simple Returns)', 'Max Drawdown']
print(performance_metrics_df[display_cols].to_string(formatters={
    'Final Net Value': '{:.4f}'.format,
    'Total Return': '{:.2%}'.format,
    'Annualized Return': '{:.2%}'.format,
    'Annualized Log Volatility': '{:.2%}'.format,
    'Annualized Simple Volatility': '{:.2%}'.format,
    'Sharpe Ratio (Simple Returns)': '{:.2f}'.format,
    'Max Drawdown': '{:.2%}'.format
}))

# --- 4. 结果展示与保存 ---
print("\n--- 步骤 4: 结果展示与保存 ---")

# 4.1 保存绩效指标表
performance_metrics_df.to_csv("portfolio_performance_metrics.csv")
print("绩效指标已保存到 portfolio_performance_metrics.csv")

# 4.2 绘制累计净值曲线图 (使用修正后的NAV)
plt.figure(figsize=(14, 7))
for strategy_name, nav_series in nav_series_all_strategies.items():
    plt.plot(nav_series.index, nav_series, label=strategy_name, alpha=0.8)

plt.title('Portfolio Strategies Cumulative Net Value (Initial Investment = 1)')
plt.xlabel('Date');
plt.ylabel('Cumulative Net Value')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
num_dates = len(weights_for_backtest)
if num_dates > 0:
    if num_dates > TRADING_DAYS_PER_YEAR * 2:
        locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10]); formatter = mdates.DateFormatter('%Y-%b')
    elif num_dates > TRADING_DAYS_PER_YEAR / 2:
        locator = mdates.MonthLocator(
            interval=max(1, num_dates // (TRADING_DAYS_PER_YEAR // 4))); formatter = mdates.DateFormatter('%Y-%m')
    else:
        locator = mdates.DayLocator(interval=max(1, num_dates // 10)); formatter = mdates.DateFormatter('%y-%m-%d')
    plt.gca().xaxis.set_major_locator(locator);
    plt.gca().xaxis.set_major_formatter(formatter)
else:
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=30, ha='right')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("portfolio_cumulative_returns.png")
print("图表已保存: portfolio_cumulative_returns.png")
plt.show()

# 4.3 绘制回撤曲线图 (基于修正后的NAV的回撤)
print("\n正在绘制回撤曲线图...")
strategies_for_drawdown_plot = ['CVaR_Optimized', 'Equal_Weight', 'BuyHold_ETF', 'BuyHold_GBI', 'BuyHold_CEA']
plt.figure(figsize=(14, 7))
for strategy_name in strategies_for_drawdown_plot:
    if strategy_name in drawdown_series_all_strategies:
        plt.plot(drawdown_series_all_strategies[strategy_name].index,
                 drawdown_series_all_strategies[strategy_name] * 100,
                 label=strategy_name, alpha=0.7)
plt.title('Portfolio Strategies Drawdown Over Time');
plt.xlabel('Date');
plt.ylabel('Drawdown (%)')
plt.legend(loc='lower left');
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
if num_dates > 0:
    plt.gca().xaxis.set_major_locator(locator)  # 使用上面定义的locator
else:
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=30, ha='right');
plt.tight_layout()
plt.savefig("portfolio_drawdown_curves.png")
print("图表已保存: portfolio_drawdown_curves.png")
plt.show()

# 4.4 绩效指标条形图 (Performance Metrics Bar Chart)
print("\nDrawing performance metrics bar chart...")
# Select key metrics to plot
metrics_to_plot_bar = ['Annualized Return', 'Annualized Simple Volatility', 'Sharpe Ratio (Simple Returns)',
                       'Max Drawdown']
plot_df_bar = performance_metrics_df[metrics_to_plot_bar].copy()

# Max Drawdown is negative, convert to positive for easier bar chart comparison
plot_df_bar['Max Drawdown'] = plot_df_bar['Max Drawdown'] * -1

# Rename columns for better chart labels (in English)
plot_df_bar.rename(columns={
    'Annualized Return': 'Annualized Return (%)',
    'Annualized Simple Volatility': 'Annualized Volatility (Simple, %)',
    'Sharpe Ratio (Simple Returns)': 'Sharpe Ratio',
    'Max Drawdown': 'Max Drawdown (Positive, %)'
}, inplace=True)

# Convert percentage metrics to numerical values for plotting (e.g., 0.1 for 10%)
# and then format the Y-axis ticks as percentages.
plot_df_bar['Annualized Return (%)'] = plot_df_bar['Annualized Return (%)'] * 100
plot_df_bar['Annualized Volatility (Simple, %)'] = plot_df_bar['Annualized Volatility (Simple, %)'] * 100
plot_df_bar['Max Drawdown (Positive, %)'] = plot_df_bar['Max Drawdown (Positive, %)'] * 100

fig_bar, axes_bar = plt.subplots(2, 2, figsize=(15, 10))
axes_bar_flat = axes_bar.flatten()  # Flatten the 2x2 array of axes for easy iteration

for i, metric_col_name in enumerate(plot_df_bar.columns):
    ax = axes_bar_flat[i]
    plot_df_bar[metric_col_name].plot(kind='bar', ax=ax, alpha=0.75, color=plt.cm.Paired(i / len(plot_df_bar.columns)))
    ax.set_title(metric_col_name, fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=9)  # Removed ha='right' as it's problematic
    ax.set_xlabel('')  # Clear auto-generated x-axis label "Strategy Name"
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Format Y-axis for percentage columns
    if '%' in metric_col_name:
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.1f}%'.format))
    else:  # For Sharpe Ratio
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))

plt.tight_layout()
plt.savefig("portfolio_performance_barchart.png")
print("Chart saved: portfolio_performance_barchart.png")
plt.show()

# 4.5 滚动夏普比率图
print("\n正在绘制滚动夏普比率图...")
rolling_window_sharpe = 60
strategies_for_rolling_sharpe = ['CVaR_Optimized', 'Equal_Weight', 'BuyHold_ETF']

if len(portfolio_daily_log_returns) > rolling_window_sharpe:
    plt.figure(figsize=(14, 7))
    for strategy_name in strategies_for_rolling_sharpe:
        if strategy_name in portfolio_daily_log_returns.columns:
            daily_log_returns_strat_rolling = portfolio_daily_log_returns[strategy_name].fillna(0)
            daily_simple_returns_strat_rolling = np.exp(daily_log_returns_strat_rolling) - 1  # 转为简单收益计算夏普

            rolling_mean_excess_simple_return = (daily_simple_returns_strat_rolling - DAILY_RISK_FREE_RATE).rolling(
                window=rolling_window_sharpe).mean()
            rolling_std_dev_simple_return = daily_simple_returns_strat_rolling.rolling(
                window=rolling_window_sharpe).std()

            rolling_sharpe_ratio = (rolling_mean_excess_simple_return * TRADING_DAYS_PER_YEAR) / \
                                   (rolling_std_dev_simple_return * np.sqrt(TRADING_DAYS_PER_YEAR))

            plt.plot(rolling_sharpe_ratio.index, rolling_sharpe_ratio,
                     label=f'{strategy_name} ({rolling_window_sharpe}d Rolling Sharpe)', alpha=0.8)

    plt.title(f'{rolling_window_sharpe}-Day Rolling Sharpe Ratio (Based on Simple Returns)')
    plt.xlabel('Date');
    plt.ylabel('Annualized Sharpe Ratio')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    if num_dates > 0:
        plt.gca().xaxis.set_major_locator(locator)  # 使用上面定义的locator
    else:
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("portfolio_rolling_sharpe_ratio.png")
    print("图表已保存: portfolio_rolling_sharpe_ratio.png")
    plt.show()
else:
    print(f"数据长度不足 {rolling_window_sharpe} 天，无法计算滚动夏普比率。")

print("\n--- 阶段六 (回溯检验与绩效评估) 完成 ---")