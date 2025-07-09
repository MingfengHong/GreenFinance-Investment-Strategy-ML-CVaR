import pandas as pd
import numpy as np
import json

# --- 0. 定义常量 ---
TRADING_DAYS_PER_YEAR = 252
MAIN_ASSETS_MODELS = {
    'etf': 'merton',  # ETF (NETF) 使用 Merton 模型
    'gbi': 'gbm',  # GBI 使用 GBM 模型
    'cea': 'merton'  # CEA 使用 Merton 模型
}
# 收益率列名，应与1_exploratory_analysis.py中的RETURN_COL_NAMES一致
RETURN_COL_NAMES = {
    'etf': 'r_etf',
    'gbi': 'r_gbi',
    'cea': 'r_cea'
}
DT = 1 / TRADING_DAYS_PER_YEAR  # 日度时间步长，用于Merton模型参数与日度收益率的转换

# --- 1. 加载阶段A1数据 ---
print("--- 步骤 1: 加载阶段A1数据 ---")
try:
    data_A1 = pd.read_csv("data_A1_multi_asset_static_est.csv", index_col='date', parse_dates=True)
    print("data_A1_multi_asset_static_est.csv 加载成功。")
except FileNotFoundError:
    print("错误: data_A1_multi_asset_static_est.csv 文件未找到。请先运行阶段一的脚本并保存该文件。")
    exit()
except Exception as e:
    print(f"读取 data_A1_multi_asset_static_est.csv 文件时发生错误: {e}")
    exit()

all_base_params = {}

# --- 2. 为各资产估计静态参数 ---
print("\n--- 步骤 2: 为各资产估计静态参数 ---")

for asset_key, model_type in MAIN_ASSETS_MODELS.items():
    print(f"\n-- 开始为资产 '{asset_key.upper()}' (模型: {model_type}) 估计参数 --")
    ret_col = RETURN_COL_NAMES.get(asset_key)
    if not ret_col or ret_col not in data_A1.columns:
        print(f"警告: 资产 '{asset_key.upper()}' 的收益率列 '{ret_col}' 在 data_A1 中未找到，跳过参数估计。")
        continue

    asset_returns_A1 = data_A1[ret_col].dropna()
    if asset_returns_A1.empty or len(asset_returns_A1) < 20:  # 至少需要一些数据点
        print(
            f"警告: 资产 '{asset_key.upper()}' 在阶段A1的有效收益率数据过少 ({len(asset_returns_A1)}条)，跳过参数估计。")
        continue

    print(f"使用 {len(asset_returns_A1)} 条来自 data_A1 的 '{ret_col}' 数据进行估计。")
    mean_daily_log_return = asset_returns_A1.mean()
    std_daily_log_return = asset_returns_A1.std()

    if model_type == 'gbm':
        print("估计 GBM 模型参数...")
        # GBM参数: mu (总预期年化收益率), sigma (年化波动率)
        # dln(S) = (mu_total - 0.5*sigma^2)dt + sigma*dWt
        # E[dln(S)/dt] = mu_total - 0.5*sigma^2
        # mu_total = E[dln(S)/dt] + 0.5*sigma^2

        # 年化日对数收益率均值 (漂移项部分)
        annualized_mean_log_return_drift_part = mean_daily_log_return * TRADING_DAYS_PER_YEAR
        # 年化波动率
        annualized_volatility = std_daily_log_return * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 总预期年化收益率 (mu_total for GBM)
        mu_total_gbm = annualized_mean_log_return_drift_part + 0.5 * (annualized_volatility ** 2)

        all_base_params[asset_key] = {
            'model_type': 'gbm',
            'mu_base': mu_total_gbm,
            'sigma_base': annualized_volatility
        }
        print(f"  GBM 静态参数 for {asset_key.upper()}:")
        print(f"    mu_base (年化总预期收益率): {mu_total_gbm:.6f}")
        print(f"    sigma_base (年化波动率): {annualized_volatility:.6f}")

    elif model_type == 'merton':
        print("估计 Merton 跳跃扩散模型参数 (启发式方法)...")
        # 启发式参数估计步骤 (与您原脚本2_static_parameter_estimation.py类似)

        # 2.1 识别潜在跳跃
        jump_threshold_multiplier = 3  # 可以调整
        jump_upper_threshold = mean_daily_log_return + jump_threshold_multiplier * std_daily_log_return
        jump_lower_threshold = mean_daily_log_return - jump_threshold_multiplier * std_daily_log_return

        potential_jumps = asset_returns_A1[
            (asset_returns_A1 > jump_upper_threshold) | (asset_returns_A1 < jump_lower_threshold)]
        normal_returns = asset_returns_A1[
            (asset_returns_A1 <= jump_upper_threshold) & (asset_returns_A1 >= jump_lower_threshold)]

        num_potential_jumps = len(potential_jumps)
        num_total_observations = len(asset_returns_A1)
        print(
            f"  识别出的潜在跳跃数量: {num_potential_jumps} / {num_total_observations} ({(num_potential_jumps / num_total_observations * 100) if num_total_observations > 0 else 0:.2f}%)")

        hat_lambda_annual = 0.0
        hat_mJ = 0.0  # 对数跳跃幅度的均值 (不是百分比)
        hat_sigmaJ = 0.0  # 对数跳跃幅度的标准差
        k_avg = 0.0  # 平均跳跃幅度百分比

        if num_potential_jumps > 0:
            hat_lambda_daily = num_potential_jumps / num_total_observations
            hat_lambda_annual = hat_lambda_daily * TRADING_DAYS_PER_YEAR
            hat_mJ = potential_jumps.mean()  # 这是对数跳跃幅度的均值 E[ln(Y)]
            if num_potential_jumps > 1:
                hat_sigmaJ = potential_jumps.std()  # 这是对数跳跃幅度的标准差 Std[ln(Y)]
            else:
                hat_sigmaJ = 1e-6  # 如果只有一个跳跃，标准差无法稳健估计，设一个极小值避免计算错误
                print("  警告: 只有一个潜在跳跃被识别，sigmaJ 的估计可能不可靠，已设为极小值。")
            # k_avg = E[Y-1] = exp(mJ + 0.5*sigmaJ^2) - 1
            k_avg = np.exp(hat_mJ + 0.5 * hat_sigmaJ ** 2) - 1
        else:
            print("  未识别到明显跳跃，lambda, mJ, sigmaJ, k_avg 将设为0或默认值。")
            # 如果没有跳跃，Merton模型退化为GBM。此时，mJ, sigmaJ, k_avg可以为0。

        # 2.2 估计扩散参数 (sigma)
        if len(normal_returns) < 2:  # 需要至少两个点来计算标准差
            print("  警告: “正常”收益率数据过少，扩散波动率估计将使用整体标准差。")
            hat_sigma_diff_daily = std_daily_log_return
        else:
            hat_sigma_diff_daily = normal_returns.std()
        hat_sigma_annual_merton = hat_sigma_diff_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 2.3 估计总预期年化收益率 (mu) for Merton
        # dln(S) = (mu_total - 0.5*sigma^2 - lambda*k_avg)dt + sigma*dWt + dJt
        # E[dln(S)]/dt = (mu_total - 0.5*sigma^2 - lambda*k_avg) + lambda*E[ln(Y)] (如果Jumps是以lnY形式加入)
        # 或者，更常见的是 E[dln(S)] = (mu_total - 0.5*sigma^2 - lambda*k_avg)dt
        # (这里的mu_total是总漂移，k_avg是补偿项，确保 E[dS/S] = mu_total*dt)
        # 我们有 mean_daily_log_return = (mu_total_annual - 0.5*hat_sigma_annual_merton^2 - hat_lambda_annual*k_avg) * DT
        # 所以, mu_total_annual = (mean_daily_log_return / DT) + 0.5*hat_sigma_annual_merton^2 + hat_lambda_annual*k_avg

        mu_total_merton = (mean_daily_log_return / DT) + 0.5 * hat_sigma_annual_merton ** 2 + hat_lambda_annual * k_avg

        all_base_params[asset_key] = {
            'model_type': 'merton',
            'mu_base': mu_total_merton,  # 总预期年化收益率
            'sigma_base': hat_sigma_annual_merton,  # 扩散部分年化波动率
            'lambda_base': hat_lambda_annual,  # 年化跳跃强度
            'mJ_base': hat_mJ,  # 平均对数跳跃幅度 E[ln(Y)]
            'sigmaJ_base': hat_sigmaJ,  # 对数跳跃幅度标准差 Std[ln(Y)]
            'k_avg_base': k_avg  # 平均跳跃幅度百分比 E[Y-1]
        }
        print(f"  Merton 静态参数 for {asset_key.upper()}:")
        print(f"    mu_base (年化总预期收益率): {mu_total_merton:.6f}")
        print(f"    sigma_base (年化扩散波动率): {hat_sigma_annual_merton:.6f}")
        print(f"    lambda_base (年化跳跃强度): {hat_lambda_annual:.6f}")
        print(f"    mJ_base (平均对数跳跃幅度): {hat_mJ:.6f}")
        print(f"    sigmaJ_base (对数跳跃幅度标准差): {hat_sigmaJ:.6f}")
        print(f"    k_avg_base (平均跳跃幅度百分比): {k_avg:.6f}")
    else:
        print(f"错误: 资产 '{asset_key.upper()}' 的模型类型 '{model_type}' 未知。")

# --- 3. 保存所有资产的静态参数 ---
print("\n--- 步骤 3: 保存所有资产的静态参数 ---")
try:
    with open('all_assets_base_params.json', 'w') as f:
        json.dump(all_base_params, f, indent=4)
    print("所有资产的基础静态参数已保存到 all_assets_base_params.json")
except Exception as e:
    print(f"保存参数到JSON文件时发生错误: {e}")

print("\n--- 阶段二 (静态参数估计) 完成 ---")