import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import jarque_bera
import matplotlib.dates as mdates

# --- 0. 定义常量 ---
# 能源期货数据的真实最后有效观测日期 (您提供的是2024/9/6，脚本中用的是前一天，这里统一为9月5日，因为9月6日的数据可能不用于当日收益率计算)
ACTUAL_FUTURES_END_DATE = pd.to_datetime('2024-09-05')  # 以实际数据为准，这里用您说的最后日期前一天作为特征截止
FUTURES_SHORT_NAMES = ['j', 'fu', 'jm']  # 能源期货简称

# 主要资产简称及列名映射
MAIN_ASSETS_KEYS = ['etf', 'gbi', 'cea']
PRICE_COL_NAMES = {
    'etf': 'etf_price',
    'gbi': 'gbi_price',
    'cea': 'cea_price'
}
RETURN_COL_NAMES = {
    'etf': 'r_etf',
    'gbi': 'r_gbi',
    'cea': 'r_cea'
}
TRADING_DAYS_PER_YEAR = 252  # 年化因子

# --- 1. 加载数据 ---
print("--- 步骤 1: 加载数据 ---")
try:
    # 假设您的数据文件名为 "multi_asset_price_data.xlsx"
    # 请确保文件名和路径正确
    raw_df = pd.read_excel("multi_asset_price_data.xlsx")
    print("数据文件加载成功。")
except FileNotFoundError:
    print("错误: multi_asset_price_data.xlsx 文件未找到。请确保文件在当前工作目录或提供正确路径。")
    exit()
except Exception as e:
    print(f"错误: 读取Excel文件时发生错误: {e}")
    exit()

print("\n原始数据预览 (前5行):")
print(raw_df.head())
print("\n原始数据预览 (后5行):")
print(raw_df.tail())

# 标准化日期列名并转换为datetime对象
if 'date' not in raw_df.columns:
    print("错误: 数据中未找到'date'列。请检查列名。")
    exit()
raw_df['date'] = pd.to_datetime(raw_df['date'])
raw_df.sort_values(by='date', inplace=True)
raw_df.set_index('date', inplace=True)
print("\n设置日期为索引后数据预览:")
print(raw_df.head())

# --- 2. 选择并处理主要资产价格序列 ---
print("\n--- 步骤 2: 选择并处理主要资产价格序列 ---")
St_dict = {}
for asset_key in MAIN_ASSETS_KEYS:
    price_col = PRICE_COL_NAMES.get(asset_key)
    if price_col and price_col in raw_df.columns:
        St_dict[asset_key] = raw_df[price_col].copy().dropna()  # 初始加载时去除完全为空的
        if St_dict[asset_key].empty:
            print(f"警告: {asset_key.upper()} 的价格列 '{price_col}' 加载后为空。")
        else:
            print(
                f"已选择 '{price_col}' 作为 {asset_key.upper()} 的价格序列，原始非空数据 {len(St_dict[asset_key])} 条。")
    else:
        print(f"错误: 未找到用于 {asset_key.upper()} 价格的列 '{price_col}'。")
        St_dict[asset_key] = pd.Series(dtype=float)  # 创建空Series以避免后续错误

# 缺失值处理 (通常在合并对齐后再严格处理)
for asset_key, St_asset in St_dict.items():
    if not St_asset.empty:
        original_nan_count = St_asset.isnull().sum()  # 在此阶段，如果上面用了dropna()，这里应该是0
        # 通常金融数据使用ffill填充周末或节假日数据
        St_asset_filled = St_asset.ffill()
        # 对于开头的NaN，如果ffill无法填充，可以考虑bfill或后续基于共同日期范围截取
        St_asset_filled = St_asset_filled.bfill()
        filled_nan_count = St_asset_filled.isnull().sum()
        St_dict[asset_key] = St_asset_filled  # 更新回字典

        if original_nan_count > 0:  # 应该在dropna后为0，除非数据中间有缺失
            print(f"{asset_key.upper()} 价格序列在选择后存在缺失值: {original_nan_count}。")
        if filled_nan_count > 0:
            print(f"警告: {asset_key.upper()} 价格序列在填充后仍有缺失值 {filled_nan_count}，请检查数据源的连续性。")
        else:
            print(f"{asset_key.upper()} 价格序列缺失值已填充。")

main_assets_price_df = pd.concat(
    [s.rename(f'St_{k}') for k, s in St_dict.items() if not s.empty],
    axis=1
)

if main_assets_price_df.empty:
    print("错误：所有主要资产价格序列均为空或加载失败。程序终止。")
    exit()

print("\n主要资产价格数据初步合并后预览:")
print(main_assets_price_df.head())
print("\n主要资产价格数据缺失值统计:")
print(main_assets_price_df.isnull().sum())

# --- 3. 处理能源期货价格数据 ---
print("\n--- 步骤 3: 处理能源期货价格数据 ---")
processed_futures_list = []
for short_name in FUTURES_SHORT_NAMES:
    if short_name in raw_df.columns:
        temp_series = raw_df[short_name].copy()
        original_future_nan_count_total = temp_series.isnull().sum()
        temp_series_filled = temp_series.ffill().bfill()
        filled_future_nan_count_total = temp_series_filled.isnull().sum()
        processed_futures_list.append(temp_series_filled.rename(short_name))
        print(
            f"已处理期货品种: {short_name}。原始数据总缺失: {original_future_nan_count_total}, 处理后剩余: {filled_future_nan_count_total}")
    else:
        print(f"警告: 未找到期货列 '{short_name}'。将跳过该品种。")

if not processed_futures_list:
    print("警告：未能处理任何能源期货数据。")
    processed_futures_df = pd.DataFrame(index=main_assets_price_df.index)
else:
    processed_futures_df = pd.concat(processed_futures_list, axis=1)

print("\n能源期货价格数据处理后预览:")
print(processed_futures_df.head())
print("\n能源期货价格数据缺失值统计:")
print(processed_futures_df.isnull().sum())

# --- 4. 合并所有价格数据并进行严格对齐 ---
print("\n--- 步骤 4: 合并所有价格数据并对齐 ---")
price_df = pd.concat([main_assets_price_df, processed_futures_df], axis=1)
common_idx_main_assets = main_assets_price_df.dropna(how='any').index
price_df = price_df.loc[common_idx_main_assets]

print("\n所有价格数据合并对齐后 (基于主资产共同日期) 预览:")
print(price_df.head())
print(price_df.tail())
print("\n合并对齐后 price_df 缺失值统计:")
print(price_df.isnull().sum())

# --- 5. 计算对数收益率 ---
print("\n--- 步骤 5: 计算对数收益率 ---")
returns_df = pd.DataFrame(index=price_df.index)
for asset_key in MAIN_ASSETS_KEYS:
    st_col_name = f'St_{asset_key}'
    ret_col_name = RETURN_COL_NAMES[asset_key]
    if st_col_name in price_df.columns:
        if (price_df[st_col_name] <= 0).any():
            print(f"警告: {st_col_name} 包含0或负数值，计算对数收益率前将替换为极小正数。")
            price_df.loc[price_df[st_col_name] <= 0, st_col_name] = 1e-9
        returns_df[ret_col_name] = np.log(price_df[st_col_name] / price_df[st_col_name].shift(1))
    else:
        print(f"警告: 价格列 {st_col_name} 不在 price_df 中，无法计算其收益率。")

for short_name in FUTURES_SHORT_NAMES:
    if short_name in price_df.columns:
        ret_col_name_futures = f'r_{short_name}'
        if (price_df[short_name] <= 0).any():
            print(f"警告: 期货 {short_name} 包含0或负数值，计算对数收益率前将替换为极小正数。")
            price_df.loc[price_df[short_name] <= 0, short_name] = 1e-9
        returns_df[ret_col_name_futures] = np.log(price_df[short_name] / price_df[short_name].shift(1))
    else:
        print(f"警告: 期货价格列 {short_name} 不在 price_df 中，无法计算其收益率。")

final_df = pd.concat([price_df, returns_df], axis=1)
required_return_cols_final = [RETURN_COL_NAMES[key] for key in MAIN_ASSETS_KEYS if
                              RETURN_COL_NAMES[key] in final_df.columns]
if not required_return_cols_final or len(required_return_cols_final) < len(MAIN_ASSETS_KEYS):
    print("错误: 一个或多个主要资产的收益率列未能成功创建或加入 final_df。")
    exit()  # 改为 exit 以确保后续步骤有数据
final_df.dropna(subset=required_return_cols_final, how='any', inplace=True)

print("\n包含价格和对数收益率的 final_df 预览:")
print(final_df.head())
print(final_df.tail())
print(f"\nfinal_df 的形状: {final_df.shape}")
print("\nfinal_df 中各列缺失值统计:")
print(final_df.isnull().sum())

# --- 6. 数据划分 ---
print("\n--- 步骤 6: 数据划分 ---")
if ACTUAL_FUTURES_END_DATE not in final_df.index:
    try:
        actual_futures_end_idx_loc = final_df.index.get_slice_bound(ACTUAL_FUTURES_END_DATE, side='right')
        if actual_futures_end_idx_loc == 0 and ACTUAL_FUTURES_END_DATE < final_df.index.min():  # 如果目标日期比所有日期都早
            ACTUAL_FUTURES_END_DATE_ADJ = final_df.index.min()
            print(
                f"警告: ACTUAL_FUTURES_END_DATE ({ACTUAL_FUTURES_END_DATE.date()}) 早于 final_df 中的最早日期。已设为最早日期: {ACTUAL_FUTURES_END_DATE_ADJ.date()}")
        elif actual_futures_end_idx_loc > 0:
            ACTUAL_FUTURES_END_DATE_ADJ = final_df.index[actual_futures_end_idx_loc - 1]
            print(
                f"注意: {ACTUAL_FUTURES_END_DATE.date()} 不是 final_df 中的交易日或超出范围，已调整为最近的前一个交易日: {ACTUAL_FUTURES_END_DATE_ADJ.date()}")
        else:  # 所有日期都小于 ACTUAL_FUTURES_END_DATE
            ACTUAL_FUTURES_END_DATE_ADJ = final_df.index.max()
            print(
                f"注意: {ACTUAL_FUTURES_END_DATE.date()} 晚于 final_df 中的所有日期，已调整为最后一个交易日: {ACTUAL_FUTURES_END_DATE_ADJ.date()}")

    except Exception as e:
        print(
            f"警告: 查找 ACTUAL_FUTURES_END_DATE ({ACTUAL_FUTURES_END_DATE.date()}) 时出错: {e}。将使用期货数据中最后一个有效日期。")
        futures_ret_cols = [f'r_{name}' for name in FUTURES_SHORT_NAMES if f'r_{name}' in final_df.columns]
        if futures_ret_cols:
            last_valid_futures_date = final_df[futures_ret_cols].dropna(how='all').index.max()
            if pd.isna(last_valid_futures_date):
                ACTUAL_FUTURES_END_DATE_ADJ = final_df.index.min()
                print("警告: 所有期货收益率数据均为空，ACTUAL_FUTURES_END_DATE_ADJ 设置为数据开始日期。")
            else:
                ACTUAL_FUTURES_END_DATE_ADJ = last_valid_futures_date
        else:
            ACTUAL_FUTURES_END_DATE_ADJ = final_df.index.max()
            print("警告: 无有效期货收益率列，ACTUAL_FUTURES_END_DATE_ADJ 设置为数据结束日期。")
else:
    ACTUAL_FUTURES_END_DATE_ADJ = ACTUAL_FUTURES_END_DATE

print(f"用于ML特征的能源期货数据有效截止日期为: {ACTUAL_FUTURES_END_DATE_ADJ.strftime('%Y-%m-%d')}")

if final_df.empty:
    print("错误: final_df 为空，无法进行数据划分。")
    exit()

start_date_overall = final_df.index[0]
# ... (阶段A1, B, C, D 的划分逻辑与您之前提供的代码相同，此处省略以减少重复) ...
# 确保这里的划分逻辑能够正确处理各种边界条件，特别是当final_df的日期范围较短时
# 阶段A1: 静态模型参数估计期 (例如，从开始的1年)
A1_end_date_target = start_date_overall + pd.DateOffset(years=1) - pd.DateOffset(days=1)
if A1_end_date_target > final_df.index[-1]: A1_end_date_target = final_df.index[-1]
A1_end_date_loc = final_df.index.get_slice_bound(A1_end_date_target, side='right')
data_A1 = pd.DataFrame()
if A1_end_date_loc > 0:
    A1_end_date = final_df.index[A1_end_date_loc - 1]
    if A1_end_date >= start_date_overall:  # 确保结束日期不早于开始日期
        data_A1_cols = [RETURN_COL_NAMES[key] for key in MAIN_ASSETS_KEYS if RETURN_COL_NAMES[key] in final_df.columns]
        if not data_A1_cols: print("警告: 阶段A1无有效的主资产收益率列。")
        data_A1 = final_df.loc[start_date_overall:A1_end_date, data_A1_cols].copy()
        print(
            f"阶段 A1 (静态参数估计): {data_A1.index.min().strftime('%Y-%m-%d')} to {data_A1.index.max().strftime('%Y-%m-%d')}, 样本数: {len(data_A1)}")
    else:
        print(
            f"警告: 阶段A1结束日期 {A1_end_date.strftime('%Y-%m-%d')} 早于开始日期 {start_date_overall.strftime('%Y-%m-%d')}。data_A1 为空。")
else:
    print("警告: 阶段A1无法划分，可能数据过少或目标结束日期早于数据起点。")

# 阶段B: 机器学习模型训练验证期
data_B = pd.DataFrame()
if not data_A1.empty and data_A1.index.max() < final_df.index.max():  # 确保A1之后还有数据
    B_start_date_temp = data_A1.index[-1] + pd.DateOffset(days=1)
    if B_start_date_temp <= final_df.index[-1]:
        B_start_date_loc = final_df.index.get_slice_bound(B_start_date_temp, side='left')
        B_start_date = final_df.index[B_start_date_loc]

        B_end_date_target = B_start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        if B_end_date_target > ACTUAL_FUTURES_END_DATE_ADJ: B_end_date_target = ACTUAL_FUTURES_END_DATE_ADJ
        if B_end_date_target > final_df.index[-1]: B_end_date_target = final_df.index[-1]

        B_end_date_loc = final_df.index.get_slice_bound(B_end_date_target, side='right')
        if B_end_date_loc > 0 and final_df.index[B_end_date_loc - 1] >= B_start_date:
            B_end_date = final_df.index[B_end_date_loc - 1]
            data_B_cols = ([RETURN_COL_NAMES[key] for key in MAIN_ASSETS_KEYS if
                            RETURN_COL_NAMES[key] in final_df.columns] +
                           [f'r_{col}' for col in FUTURES_SHORT_NAMES if f'r_{col}' in final_df.columns])
            if not data_B_cols: print("警告: 阶段B无有效的收益率列。")
            data_B = final_df.loc[B_start_date:B_end_date, data_B_cols].copy()
            futures_returns_for_B = [f'r_{col}' for col in FUTURES_SHORT_NAMES if f'r_{col}' in data_B.columns]
            if futures_returns_for_B: data_B.dropna(subset=futures_returns_for_B, how='any', inplace=True)
            if not data_B.empty:
                print(
                    f"阶段 B (ML训练/验证): {data_B.index.min().strftime('%Y-%m-%d')} to {data_B.index.max().strftime('%Y-%m-%d')}, 样本数: {len(data_B)}")
            else:
                print("警告: 阶段 B 数据为空或时间范围不合法。")
        else:
            print("警告: 阶段 B 结束日期早于开始日期或无法划分。")
    else:
        print("警告: 阶段 B 的起始日期超出数据范围。")
else:
    print("警告: 因阶段 A1 数据为空或已是数据末尾，无法定义阶段 B。")

# 阶段C: 动态模型/组合优化评估期
data_C = pd.DataFrame()
if not data_B.empty and data_B.index.max() < ACTUAL_FUTURES_END_DATE_ADJ and data_B.index.max() < final_df.index.max():
    C_start_date_temp = data_B.index[-1] + pd.DateOffset(days=1)
    if C_start_date_temp <= ACTUAL_FUTURES_END_DATE_ADJ and C_start_date_temp <= final_df.index[-1]:
        C_start_date_loc = final_df.index.get_slice_bound(C_start_date_temp, side='left')
        C_start_date = final_df.index[C_start_date_loc]
        C_end_date = ACTUAL_FUTURES_END_DATE_ADJ
        if C_end_date > final_df.index.max(): C_end_date = final_df.index.max()  # 不能超过final_df范围

        if C_end_date >= C_start_date:
            data_C_price_cols = [f'St_{key}' for key in MAIN_ASSETS_KEYS if f'St_{key}' in final_df.columns]
            data_C_return_cols = [RETURN_COL_NAMES[key] for key in MAIN_ASSETS_KEYS if
                                  RETURN_COL_NAMES[key] in final_df.columns]
            data_C_futures_price_cols = [col for col in FUTURES_SHORT_NAMES if col in final_df.columns]
            data_C_futures_return_cols = [f'r_{col}' for col in FUTURES_SHORT_NAMES if f'r_{col}' in final_df.columns]
            data_C_all_cols = list(
                set(data_C_price_cols + data_C_return_cols + data_C_futures_price_cols + data_C_futures_return_cols))
            data_C_all_cols_exist = [col for col in data_C_all_cols if col in final_df.columns]
            if not data_C_all_cols_exist: print("警告: 阶段C无有效的列。")
            data_C = final_df.loc[C_start_date:C_end_date, data_C_all_cols_exist].copy()
            # 对于阶段C，ML会用到期货特征，所以期货相关列的NaN需要处理，但特征通常是滞后的，所以原始价格和收益率列本身在ACTUAL_FUTURES_END_DATE_ADJ之前应该有值
            # 如果dropna过于严格可能导致数据为空，这里假设特征工程阶段会处理滞后特征的NaN
            # data_C.dropna(subset=data_C_futures_return_cols + data_C_futures_price_cols, how='any', inplace=True) # 过于严格可能导致数据为空
            if not data_C.empty:
                print(
                    f"阶段 C (动态模型/组合评估): {data_C.index.min().strftime('%Y-%m-%d')} to {data_C.index.max().strftime('%Y-%m-%d')}, 样本数: {len(data_C)}")
            else:
                print("警告: 阶段 C 数据为空或时间范围不合法。")
        else:
            print("警告: 阶段 C 结束日期早于开始日期。")
    else:
        print("警告: 阶段 C 的起始日期超出能源期货数据有效范围或整体数据范围。")
else:
    print("警告: 因阶段 B 数据为空或已是数据末尾/期货特征截止日期，无法定义阶段 C。")

# 阶段D: 静态模型/组合优化延伸评估期
data_D = pd.DataFrame()
if ACTUAL_FUTURES_END_DATE_ADJ < final_df.index[-1]:
    D_start_date_temp = ACTUAL_FUTURES_END_DATE_ADJ + pd.DateOffset(days=1)
    if D_start_date_temp <= final_df.index[-1]:
        D_start_date_loc = final_df.index.get_slice_bound(D_start_date_temp, side='left')
        D_start_date = final_df.index[D_start_date_loc]
        D_end_date = final_df.index[-1]
        if D_end_date >= D_start_date:
            data_D_price_cols = [f'St_{key}' for key in MAIN_ASSETS_KEYS if f'St_{key}' in final_df.columns]
            data_D_return_cols = [RETURN_COL_NAMES[key] for key in MAIN_ASSETS_KEYS if
                                  RETURN_COL_NAMES[key] in final_df.columns]
            data_D_all_cols = list(set(data_D_price_cols + data_D_return_cols))
            data_D_all_cols_exist = [col for col in data_D_all_cols if col in final_df.columns]
            if not data_D_all_cols_exist: print("警告: 阶段D无有效的列。")
            data_D = final_df.loc[D_start_date:D_end_date, data_D_all_cols_exist].copy()
            if not data_D.empty:
                print(
                    f"阶段 D (静态模型/组合延伸评估): {data_D.index.min().strftime('%Y-%m-%d')} to {data_D.index.max().strftime('%Y-%m-%d')}, 样本数: {len(data_D)}")
            else:
                print("警告: 阶段 D 数据为空或时间范围不合法。")
        else:
            print("警告: 阶段 D 结束日期早于开始日期。")
    else:
        print("阶段 D 无数据，因其起始日期超出数据范围。")
else:
    print("阶段 D 无数据，因数据不晚于最后有效期货数据，或期货数据截止日设置有误。")

# --- 7. 探索性数据分析 (EDA) ---
print("\n--- 步骤 7: 探索性数据分析 (EDA) ---")

# 7.1 主要资产价格与收益率图表
if not final_df.empty:
    num_main_assets = len(MAIN_ASSETS_KEYS)
    fig_height = 5 * num_main_assets
    fig, axes = plt.subplots(num_main_assets, 2, figsize=(15, fig_height),
                             sharex='col')  # sharex for price and return plots
    if num_main_assets == 1:  # 如果只有一个主资产，axes不是二维数组
        axes = np.array([axes])  # 转换为二维数组以便统一处理

    for i, asset_key in enumerate(MAIN_ASSETS_KEYS):
        st_col = f'St_{asset_key}'
        ret_col = RETURN_COL_NAMES[asset_key]

        # 价格图
        ax_price = axes[i, 0]
        if st_col in final_df.columns:
            final_df[st_col].plot(ax=ax_price, title=f'{asset_key.upper()} Spot Price')
            ax_price.set_ylabel('Price')
            ax_price.grid(True)
        else:
            ax_price.set_title(f'{asset_key.upper()} Spot Price (No Data)')

        # 收益率图
        ax_return = axes[i, 1]
        if ret_col in final_df.columns and not final_df[ret_col].dropna().empty:
            final_df[ret_col].plot(ax=ax_return, title=f'{asset_key.upper()} Log Returns')
            ax_return.set_ylabel('Log Return')
            ax_return.grid(True)
        else:
            ax_return.set_title(f'{asset_key.upper()} Log Returns (No Data)')

        if i < num_main_assets - 1:  # 只在非最后一个子图对上隐藏x轴标签
            ax_price.set_xlabel('')
            ax_return.set_xlabel('')
        else:
            ax_price.set_xlabel('Date')
            ax_return.set_xlabel('Date')

    plt.tight_layout()
    plt.savefig("main_assets_prices_returns.png")
    print("图表已保存: main_assets_prices_returns.png")
    plt.show()

# 7.2 主要资产收益率统计特性
print("\n--- 主要资产收益率统计特性 ---")
for asset_key in MAIN_ASSETS_KEYS:
    ret_col = RETURN_COL_NAMES[asset_key]
    if ret_col in final_df.columns and not final_df[ret_col].dropna().empty:
        print(f"\n--- {asset_key.upper()} 对数收益率 ({ret_col}) ---")
        desc_stats = final_df[ret_col].describe()
        desc_stats['skewness'] = final_df[ret_col].skew()
        desc_stats['kurtosis'] = final_df[ret_col].kurtosis()  # Pandas kurtosis is excess kurtosis
        print(desc_stats)

        fig_dist, axes_dist = plt.subplots(1, 2, figsize=(14, 5))
        fig_dist.suptitle(f'{asset_key.upper()} Log Returns Analysis', fontsize=16)

        sns.histplot(final_df[ret_col].dropna(), kde=True, bins=50, ax=axes_dist[0])
        axes_dist[0].set_title('Histogram with KDE')
        axes_dist[0].set_xlabel('Log Return')

        sm.qqplot(final_df[ret_col].dropna(), line='s', ax=axes_dist[1])
        axes_dist[1].set_title('QQ-Plot vs Normal Distribution')
        axes_dist[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{asset_key}_returns_dist.png")
        print(f"图表已保存: {asset_key}_returns_dist.png")
        plt.show()

        if len(final_df[ret_col].dropna()) > 7:
            jb_stat, jb_p_value = jarque_bera(final_df[ret_col].dropna())
            print(f"Jarque-Bera 检验统计量: {jb_stat:.4f}, p值: {jb_p_value:.4f}")
            if jb_p_value < 0.05:
                print("Jarque-Bera检验表明该收益率序列不服从正态分布 (p < 0.05)。")
            else:
                print("Jarque-Bera检验表明该收益率序列可能服从正态分布 (p >= 0.05)。")
        else:
            print(f"{asset_key.upper()} 收益率数据点过少 ({len(final_df[ret_col].dropna())})，无法进行Jarque-Bera检验。")

# 7.3 能源期货收益率图表
futures_returns_cols_plot = [f'r_{col}' for col in FUTURES_SHORT_NAMES if f'r_{col}' in final_df.columns]
if futures_returns_cols_plot:
    valid_futures_returns_for_plot = final_df[futures_returns_cols_plot].dropna(how='all')
    if not valid_futures_returns_for_plot.empty:
        num_plots_futures = len(futures_returns_cols_plot)
        fig_futures, axes_futures = plt.subplots(num_plots_futures, 1, figsize=(14, 2.5 * num_plots_futures),
                                                 sharex=True)
        if num_plots_futures == 1: axes_futures = [axes_futures]  # Make it iterable
        fig_futures.suptitle('Energy Futures Log Returns', fontsize=16)
        for i, col_name in enumerate(futures_returns_cols_plot):
            if not final_df[col_name].dropna().empty:
                final_df[col_name].plot(ax=axes_futures[i], title=f'{col_name} Log Returns')
                axes_futures[i].set_ylabel('Log Return')
                axes_futures[i].grid(True)
        axes_futures[-1].set_xlabel('Date')  # Set x-label only for the last subplot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("energy_futures_returns.png")
        print("图表已保存: energy_futures_returns.png")
        plt.show()

# 7.4 扩展的相关系数矩阵
print("\n--- 扩展的相关系数矩阵 ---")
corr_start_date = data_B.index.min() if not data_B.empty else final_df.index.min()
corr_end_date = ACTUAL_FUTURES_END_DATE_ADJ
if corr_start_date < final_df.index.min(): corr_start_date = final_df.index.min()
if corr_end_date > final_df.index.max(): corr_end_date = final_df.index.max()

if corr_start_date <= corr_end_date:
    cols_for_correlation = (
            [RETURN_COL_NAMES[key] for key in MAIN_ASSETS_KEYS if RETURN_COL_NAMES[key] in final_df.columns] +
            [f'r_{col}' for col in FUTURES_SHORT_NAMES if f'r_{col}' in final_df.columns])
    cols_for_correlation_exist = [col for col in cols_for_correlation if col in final_df.columns]

    if len(cols_for_correlation_exist) > 1:
        correlation_df_full = final_df.loc[corr_start_date:corr_end_date, cols_for_correlation_exist].copy()
        correlation_df_full.dropna(how='any', inplace=True)
        if not correlation_df_full.empty and len(correlation_df_full.columns) > 1:
            correlation_matrix_full = correlation_df_full.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix_full, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
                        annot_kws={"size": 8})
            plt.title(
                f'Log Returns Correlation Matrix ({correlation_df_full.index.min().strftime("%Y-%m-%d")} to {correlation_df_full.index.max().strftime("%Y-%m-%d")})')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig("full_correlation_matrix.png")
            print("图表已保存: full_correlation_matrix.png")
            plt.show()
            print(
                f"\n完整对数收益率相关系数矩阵 (基于 {correlation_df_full.index.min().strftime('%Y-%m-%d')} 到 {correlation_df_full.index.max().strftime('%Y-%m-%d')}):")
            print(correlation_matrix_full)
            main_asset_return_cols_exist = [RETURN_COL_NAMES[key] for key in MAIN_ASSETS_KEYS if
                                            RETURN_COL_NAMES[key] in correlation_matrix_full.columns]
            if len(main_asset_return_cols_exist) > 1:
                print("\n主要资产间收益率相关性:")
                print(correlation_matrix_full.loc[main_asset_return_cols_exist, main_asset_return_cols_exist])
        else:
            print("\n无法计算扩展相关系数矩阵，因数据不足或选定列不存在。")
    else:
        print("\n无法计算扩展相关系数矩阵，因无足够列进行分析。")
else:
    print("\n相关性分析的起止日期不合法或 data_B 为空，无法计算扩展相关系数矩阵。")

# --- 新增可视化：7.5 主要资产收益率的滚动相关性 ---
print("\n--- 主要资产收益率的滚动相关性 (60天窗口) ---")
main_asset_returns_df = final_df[
    [RETURN_COL_NAMES[key] for key in MAIN_ASSETS_KEYS if RETURN_COL_NAMES[key] in final_df.columns]].copy()
main_asset_returns_df.dropna(inplace=True)  # 确保没有NaN影响滚动计算

if len(main_asset_returns_df.columns) >= 2 and len(main_asset_returns_df) > 60:  # 需要至少两个资产和足够数据
    rolling_corr_window = 60
    asset_pairs = []
    if 'r_etf' in main_asset_returns_df.columns and 'r_gbi' in main_asset_returns_df.columns:
        asset_pairs.append(('r_etf', 'r_gbi'))
    if 'r_etf' in main_asset_returns_df.columns and 'r_cea' in main_asset_returns_df.columns:
        asset_pairs.append(('r_etf', 'r_cea'))
    if 'r_gbi' in main_asset_returns_df.columns and 'r_cea' in main_asset_returns_df.columns:
        asset_pairs.append(('r_gbi', 'r_cea'))

    if asset_pairs:
        plt.figure(figsize=(14, 3 * len(asset_pairs)))
        plt.suptitle(f'{rolling_corr_window}-Day Rolling Correlations', fontsize=16)
        for i, (asset1, asset2) in enumerate(asset_pairs):
            rolling_correlation = main_asset_returns_df[asset1].rolling(window=rolling_corr_window).corr(
                main_asset_returns_df[asset2])
            ax = plt.subplot(len(asset_pairs), 1, i + 1)
            rolling_correlation.plot(ax=ax, title=f'Rolling Correlation: {asset1.upper()} vs {asset2.upper()}')
            ax.set_ylabel('Correlation')
            ax.grid(True)
            if i < len(asset_pairs) - 1: ax.set_xlabel('')  # 隐藏非最后一个子图的x轴标签

        plt.xlabel('Date')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("rolling_correlations.png")
        print("图表已保存: rolling_correlations.png")
        plt.show()
    else:
        print("主要资产不足两个，无法计算滚动相关性。")
else:
    print("主要资产数据不足或少于两个，无法计算滚动相关性。")

# --- 新增可视化：7.6 主要资产收益率的配对散点图 ---
print("\n--- 主要资产收益率的配对散点图 ---")
if len(main_asset_returns_df.columns) > 1:  # 需要至少两个资产
    # 为了更好的可视化，可以对极端值进行一些处理，或者直接绘图。例如，可以取分位数截尾来减少极端值对散点图刻度的影响，但这会改变原始数据，需谨慎
    # temp_plot_df = main_asset_returns_df.apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)), axis=0)
    temp_plot_df = main_asset_returns_df.copy()
    temp_plot_df.columns = [col.upper() for col in temp_plot_df.columns]  # 列名改大写，图例好看

    pair_plot_fig = sns.pairplot(temp_plot_df.dropna(), kind='scatter', diag_kind='kde',
                                 plot_kws={'alpha': 0.5, 's': 20}, diag_kws={'fill': True, 'alpha': 0.6})
    pair_plot_fig.fig.suptitle('Pair Plot of Main Asset Log Returns', y=1.02, fontsize=16)
    pair_plot_fig.savefig("pairplot_main_asset_returns.png")
    print("图表已保存: pairplot_main_asset_returns.png")
    plt.show()
else:
    print("主要资产数量不足两个，无法绘制配对散点图。")

# --- 8. 保存处理后的数据 ---
print("\n--- 步骤 8: 保存处理后的数据 ---")
try:
    final_df.to_csv("final_data_for_modeling_multi_asset.csv")
    print("final_df 已保存到 final_data_for_modeling_multi_asset.csv")
    if not data_A1.empty: data_A1.to_csv("data_A1_multi_asset_static_est.csv")
    if not data_B.empty: data_B.to_csv("data_B_multi_asset_ml_train_val.csv")
    if not data_C.empty: data_C.to_csv("data_C_multi_asset_dynamic_eval.csv")
    if not data_D.empty: data_D.to_csv("data_D_multi_asset_static_ext_eval.csv")
    print("各阶段数据已准备好保存。")
except Exception as e:
    print(f"保存文件时发生错误: {e}")

print("\n--- 阶段一 (数据准备与EDA) 完成 ---")