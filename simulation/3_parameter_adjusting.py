import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score  # 新增导入
import joblib
import json
import matplotlib.pyplot as plt  # 新增导入
import seaborn as sns  # 新增导入

# --- 0. 定义常量与加载基础参数 ---
print("--- 阶段三：基于能源期货价格的参数动态调整 (多资产) ---")
FUTURES_RETURNS_COLS = ['r_j', 'r_fu', 'r_jm']
H_FORWARD_LOOKING = 20
TRADING_DAYS_PER_YEAR = 252
FEATURE_LAGS = [1, 3, 5]
ROLLING_WINDOW_SIZE = 5
ML_ASSETS = ['etf', 'cea']

RETURN_COL_NAMES = {
    'etf': 'r_etf',
    'gbi': 'r_gbi',
    'cea': 'r_cea'
}

try:
    with open('all_assets_base_params.json', 'r') as f:
        all_base_params = json.load(f)
    print("所有资产的基础静态参数 (all_assets_base_params.json) 加载成功。")
except FileNotFoundError:
    print("错误: all_assets_base_params.json 文件未找到。请先运行阶段二的脚本。")
    exit()
except Exception as e:
    print(f"读取 all_assets_base_params.json 文件时发生错误: {e}")
    exit()

try:
    data_B = pd.read_csv("data_B_multi_asset_ml_train_val.csv", index_col='date', parse_dates=True)
    if data_B.empty:
        raise ValueError("阶段 B 数据 (data_B_multi_asset_ml_train_val.csv) 为空。")
    print(
        f"\n阶段 B 数据加载成功，共 {len(data_B)} 条，日期从 {data_B.index.min().strftime('%Y-%m-%d')} 到 {data_B.index.max().strftime('%Y-%m-%d')}")
except FileNotFoundError:
    print("错误: data_B_multi_asset_ml_train_val.csv 文件未找到。")
    exit()
except Exception as e:
    print(f"读取 data_B_multi_asset_ml_train_val.csv 文件时发生错误: {e}")
    exit()

# --- 1. 构建特征变量 X (能源期货特征，对所有ML资产通用) ---
print("\n--- 步骤 1: 构建通用特征变量 X (能源期货滞后及滚动收益率) ---")
data_B_ml = data_B.copy()
feature_cols = []
for col in FUTURES_RETURNS_COLS:
    if col in data_B_ml.columns:
        for lag in FEATURE_LAGS:
            feature_name = f'lag{lag}_{col}'
            data_B_ml[feature_name] = data_B_ml[col].shift(lag)
            feature_cols.append(feature_name)
        rolling_mean_col_name = f'roll_mean{ROLLING_WINDOW_SIZE}_{col}'
        data_B_ml[rolling_mean_col_name] = data_B_ml[col].shift(1).rolling(window=ROLLING_WINDOW_SIZE).mean()
        feature_cols.append(rolling_mean_col_name)
    else:
        print(f"警告: 能源期货收益率列 {col} 不在 data_B 中，无法创建其相关特征。")

if not feature_cols:
    print("错误: 未能构建任何有效的能源期货特征列。程序终止。")
    exit()
print(f"构建的能源期货特征列: {feature_cols}")

# --- 1b. 在 final_df 上预计算所有特征 (用于后续阶段的预测) ---
print("\n--- 步骤 1b: 在完整数据集上预计算能源期货特征以备后续预测使用 ---")
try:
    final_df_for_features = pd.read_csv("final_data_for_modeling_multi_asset.csv", index_col='date', parse_dates=True)
    if final_df_for_features.empty:
        raise ValueError("final_data_for_modeling_multi_asset.csv 为空。")

    missing_orig_futures_cols = [col for col in FUTURES_RETURNS_COLS if col not in final_df_for_features.columns]
    if missing_orig_futures_cols:
        print(
            f"错误: final_data_for_modeling_multi_asset.csv 缺少以下原始能源期货收益率列: {missing_orig_futures_cols}，无法生成特征。")
        exit()

    for original_col_name in FUTURES_RETURNS_COLS:
        for lag in FEATURE_LAGS:
            lag_feature_name = f'lag{lag}_{original_col_name}'
            final_df_for_features[lag_feature_name] = final_df_for_features[original_col_name].shift(lag)
        roll_mean_feature_name = f'roll_mean{ROLLING_WINDOW_SIZE}_{original_col_name}'
        final_df_for_features[roll_mean_feature_name] = \
            final_df_for_features[original_col_name].shift(1).rolling(window=ROLLING_WINDOW_SIZE).mean()

    # 将保存操作移出循环，确保所有特征计算完毕后一次性保存
    final_df_for_features.to_csv("final_data_for_modeling_multi_asset.csv")
    print("final_df_for_features (包含预计算特征) 已更新并保存到 final_data_for_modeling_multi_asset.csv")
    print("在 final_df_for_features 上预计算能源期货特征完成。")

except Exception as e:
    print(f"加载或处理 final_data_for_modeling_multi_asset.csv 以预计算特征时失败: {e}")
    exit()

# --- 2. 循环为每个需要ML的资产训练模型 ---
trained_ml_models_info = {}
tscv = TimeSeriesSplit(n_splits=5)  # 定义一次tscv供后续使用

for asset_key in ML_ASSETS:
    print(f"\n--- 开始为资产 '{asset_key.upper()}' 训练ML模型 ---")
    asset_return_col = RETURN_COL_NAMES.get(asset_key)
    if not asset_return_col or asset_return_col not in data_B_ml.columns:
        print(f"警告: 资产 '{asset_key.upper()}' 的收益率列 '{asset_return_col}' 不在 data_B_ml 中，无法训练模型。")
        continue

    target_col_name = f'target_mu_{asset_key}_dyn_annualized'
    data_B_ml[target_col_name] = data_B_ml[asset_return_col].rolling(window=H_FORWARD_LOOKING).mean().shift(
        -H_FORWARD_LOOKING) * TRADING_DAYS_PER_YEAR

    current_asset_train_df = data_B_ml.dropna(subset=feature_cols + [target_col_name])
    if current_asset_train_df.empty:
        print(f"错误: 为资产 '{asset_key.upper()}' 构建目标变量或特征后，训练数据为空。")
        continue

    X_train_df_asset = current_asset_train_df[feature_cols].copy()
    y_train_asset = current_asset_train_df[target_col_name].copy()
    print(f"为资产 '{asset_key.upper()}' 构建目标变量和特征完成，共 {len(X_train_df_asset)} 个有效训练样本。")

    scaler = StandardScaler()  # 每次循环为当前资产数据重新初始化和fit scaler
    X_train_scaled_asset = scaler.fit_transform(X_train_df_asset)

    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_param_grid = {
        'n_estimators': [50, 100], 'max_depth': [3, 5, 7],
        'min_samples_split': [10, 20], 'min_samples_leaf': [5, 10]
    }

    print(f"开始为 '{asset_key.upper()}' 进行 GridSearchCV (随机森林)...")
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid,
                                  cv=tscv, scoring='r2', n_jobs=-1, verbose=0)
    grid_search_rf.fit(X_train_scaled_asset, y_train_asset)

    best_rf_estimator = grid_search_rf.best_estimator_
    r2_train_asset = best_rf_estimator.score(X_train_scaled_asset, y_train_asset)

    print(f"资产 '{asset_key.upper()}' 的随机森林模型训练完成。")
    print(f"  最优参数: {grid_search_rf.best_params_}")
    print(f"  训练集 R^2: {r2_train_asset:.4f}")

    final_pipeline_asset = Pipeline([
        ('scaler', scaler),
        ('regressor', best_rf_estimator)
    ])

    model_filename = f'dynamic_mu_{asset_key}_model.joblib'
    model_to_save_asset = {
        'pipeline': final_pipeline_asset,
        'name': f'RandomForest_{asset_key.upper()}',
        'feature_names': feature_cols,
        'r2_train': r2_train_asset,
        'best_params': grid_search_rf.best_params_  # 保存最优参数
    }
    joblib.dump(model_to_save_asset, model_filename)
    print(f"资产 '{asset_key.upper()}' 的动态漂移项预测模型已保存到 {model_filename}")
    trained_ml_models_info[asset_key] = model_to_save_asset

    # 2a. 特征重要性条形图
    if hasattr(best_rf_estimator, 'feature_importances_'):
        importances = pd.Series(best_rf_estimator.feature_importances_, index=feature_cols)
        sorted_importances = importances.sort_values(ascending=True)
        plt.figure(figsize=(10, len(feature_cols) * 0.3 + 1))  # 动态调整高度
        sorted_importances.plot(kind='barh')
        plt.title(f'Feature Importances for {asset_key.upper()} Dynamic Mu Model')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f"feature_importance_{asset_key}.png")
        print(f"图表已保存: feature_importance_{asset_key}.png")
        plt.show()

    # 2b. 动态漂移项时间序列图 (基于训练集预测)
    # 使用已经fit好的pipeline在训练特征上（已缩放的）进行预测
    # 或者更直接：用pipeline在未缩放的X_train_df_asset上预测
    predicted_mu_dyn_train = final_pipeline_asset.predict(X_train_df_asset)

    plt.figure(figsize=(14, 7))
    plt.plot(X_train_df_asset.index, predicted_mu_dyn_train, label=f'Predicted Dynamic Mu ({asset_key.upper()})',
             alpha=0.7)
    static_mu_asset = all_base_params.get(asset_key, {}).get('mu_base', np.nan)
    if not pd.isna(static_mu_asset):
        plt.axhline(static_mu_asset, color='red', linestyle='--',
                    label=f'Static Mu_base ({asset_key.upper()}) = {static_mu_asset:.4f}')
    plt.title(f'Predicted Dynamic Mu vs Static Mu for {asset_key.upper()} (Training Period B)')
    plt.xlabel('Date')
    plt.ylabel('Annualized Mu')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"dynamic_mu_timeseries_{asset_key}.png")
    print(f"图表已保存: dynamic_mu_timeseries_{asset_key}.png")
    plt.show()

    # 2c. 实际 vs. 预测散点图 (基于交叉验证)
    print(f"为资产 '{asset_key.upper()}' 生成交叉验证的实际 vs. 预测散点图...")
    # 使用 cross_val_predict 获取交叉验证的预测结果
    # 需要重新构建一个包含最优参数的模型实例，并使用Pipeline
    cv_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 每次CV折叠内部会重新fit scaler
        ('regressor', RandomForestRegressor(**grid_search_rf.best_params_, random_state=42, n_jobs=-1))
    ])
    try:
        # cross_val_predict 直接在原始X_train_df_asset上操作，Pipeline内部会处理缩放
        y_pred_cv = cross_val_predict(cv_pipeline, X_train_df_asset, y_train_asset, cv=tscv, n_jobs=-1)

        cv_r2 = r2_score(y_train_asset, y_pred_cv)
        print(f"  交叉验证 R^2 for '{asset_key.upper()}': {cv_r2:.4f}")

        plt.figure(figsize=(8, 8))
        plt.scatter(y_train_asset, y_pred_cv, alpha=0.5)
        # 添加 y=x 参考线
        min_val = min(y_train_asset.min(), y_pred_cv.min())
        max_val = max(y_train_asset.max(), y_pred_cv.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
        plt.title(f'Actual vs. Predicted Dynamic Mu for {asset_key.upper()} (Cross-Validated)')
        plt.xlabel('Actual Future Avg. Annualized Return (Target)')
        plt.ylabel('Predicted Future Avg. Annualized Return (CV)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"actual_vs_pred_cv_{asset_key}.png")
        print(f"图表已保存: actual_vs_pred_cv_{asset_key}.png")
        plt.show()
    except Exception as e_cv:
        print(f"  为资产 '{asset_key.upper()}' 生成CV散点图时发生错误: {e_cv}")


# --- 3. 定义动态参数预测函数 (通用) ---
# (与您上一版本的脚本相同，此处省略以节省空间)
def predict_dynamic_mu_for_asset(features_for_single_prediction_df, asset_model_artifacts, base_mu_val):
    if not asset_model_artifacts or 'pipeline' not in asset_model_artifacts or 'feature_names' not in asset_model_artifacts:
        return base_mu_val
    model_pipeline = asset_model_artifacts['pipeline']
    trained_feature_names_list = asset_model_artifacts['feature_names']
    if features_for_single_prediction_df.empty:
        return base_mu_val
    try:
        features_ordered = features_for_single_prediction_df[trained_feature_names_list]
        if features_ordered.isnull().values.any().any():
            return base_mu_val
        predicted_mu_dyn_array = model_pipeline.predict(features_ordered)
        return predicted_mu_dyn_array[0] if len(predicted_mu_dyn_array) > 0 else base_mu_val
    except KeyError as e:
        print(f"警告: 动态mu预测时发生KeyError (特征缺失或不匹配: {e})。回退到base_mu。")
        return base_mu_val
    except Exception as e:
        print(
            f"警告: 动态mu预测过程中发生错误 ({features_for_single_prediction_df.index[0].date()}): {e}。回退到base_mu。")
        return base_mu_val


# --- 4. 预测函数使用示例 ---
# (与您上一版本的脚本相同，此处省略以节省空间)
if final_df_for_features is not None and not data_B.empty and trained_ml_models_info:
    first_asset_key_for_example = next(iter(trained_ml_models_info))
    last_day_in_B = data_B.index[-1]

    # 从 all_base_params 中安全地获取 ACTUAL_FUTURES_END_DATE_ADJ_STR
    # 如果1_exploratory_analysis.py中已将其保存到all_assets_base_params.json
    # 否则使用默认值。
    actual_futures_end_date_adj_str = '2024-09-05'  # 默认值
    if 'ACTUAL_FUTURES_END_DATE_ADJ_STR' in all_base_params:
        actual_futures_end_date_adj_str = all_base_params['ACTUAL_FUTURES_END_DATE_ADJ_STR']
    elif 'etf' in all_base_params and 'ACTUAL_FUTURES_END_DATE_ADJ_STR' in all_base_params['etf']:  # 兼容旧格式
        actual_futures_end_date_adj_str = all_base_params['etf']['ACTUAL_FUTURES_END_DATE_ADJ_STR']

    potential_example_dates = final_df_for_features.loc[
        (final_df_for_features.index > last_day_in_B) &
        (final_df_for_features.index <= pd.to_datetime(actual_futures_end_date_adj_str))
        ].dropna(subset=feature_cols).index

    if not potential_example_dates.empty:
        example_prediction_target_date = potential_example_dates[0]
        print(f"\n动态漂移项预测函数使用示例 (预测 {example_prediction_target_date.strftime('%Y-%m-%d')} 的mu):")
        if example_prediction_target_date in final_df_for_features.index:
            if set(feature_cols).issubset(final_df_for_features.columns):
                example_features_row = final_df_for_features.loc[[example_prediction_target_date], feature_cols].copy()
                if not example_features_row.empty:
                    print("    构造的用于预测的能源期货特征行:")
                    print(example_features_row)
                    for asset_key_ex, model_artifacts_ex in trained_ml_models_info.items():
                        if asset_key_ex in all_base_params and 'mu_base' in all_base_params[asset_key_ex]:
                            base_mu_val_ex = all_base_params[asset_key_ex]['mu_base']
                            predicted_mu_example = predict_dynamic_mu_for_asset(example_features_row,
                                                                                model_artifacts_ex,
                                                                                base_mu_val_ex)
                            print(
                                f"  资产 '{asset_key_ex.upper()}' 预测得到的动态 mu_dyn (年化): {predicted_mu_example:.6f}")
                            print(
                                f"    对比的基础静态 mu_base (年化) for '{asset_key_ex.upper()}': {base_mu_val_ex:.6f}")
                        else:
                            print(f"    资产 '{asset_key_ex.upper()}' 的基础mu_base未在all_base_params中找到。")
                else:
                    print(f"    未能为示例预测日期 {example_prediction_target_date.strftime('%Y-%m-%d')} 提取到特征行。")
            else:
                print(f"    错误: final_df_for_features 中缺少模型训练所需的全部能源期货特征列。期望: {feature_cols}")
        else:
            print(
                f"    示例预测日期 {example_prediction_target_date.strftime('%Y-%m-%d')} 不在包含预计算特征的 final_df_for_features 中。")
    else:
        print("\n在 data_B 之后没有更多合适的日期可用于示例预测（需要完整特征且在期货数据有效期内）。")
else:
    if 'final_df_for_features' not in locals() or final_df_for_features is None: print(
        "\n无法进行示例预测，因为 final_df_for_features 未加载或为空。")
    if data_B.empty: print("\n无法进行示例预测，因为 data_B 为空。")
    if not trained_ml_models_info: print("\n无法进行示例预测，因为没有成功训练的ML模型。")

print("\n--- 阶段三 (ML参数动态化) 完成 ---")