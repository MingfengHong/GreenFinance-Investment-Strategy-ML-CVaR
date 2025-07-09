import pandas as pd
import os

# 更新后的 Excel 文件名列表
file_names_full = [
    "etf_price.xlsx",
    "gbi_price.xlsx",
    "cea_price.xlsx"
    # 您可以根据需要添加更多文件名
]

# 初始化合并后的 DataFrame
merged_df = None

print("Starting data processing for XLSX files...")

try:
    for file_name_full in file_names_full:
        print(f"Processing file: {file_name_full}...")

        # 检查文件是否存在
        if not os.path.exists(file_name_full):
            print(
                f"Error: File not found - {file_name_full}. Please ensure it's in the same directory as the script. Skipping this file.")
            continue

        # 读取 Excel 文件 (默认读取第一个sheet)
        try:
            # 如果您的数据不在第一个sheet，您可能需要指定 sheet_name, 例如:
            # df = pd.read_excel(file_name_full, sheet_name='YourSheetName')
            df = pd.read_excel(file_name_full)
        except Exception as e:
            print(f"Error reading {file_name_full}: {e}. Skipping this file.")
            continue

        # 从文件名提取前缀
        # 例如从 "etf_price.xlsx" 提取 "etf"
        base_name = file_name_full.split('.')[0]  # 移除 ".xlsx"
        if '_' in base_name:
            prefix = base_name.split('_')[0].lower()
        else:  # 如果文件名不含下划线, 例如 "etf.xlsx"，则使用 "etf"
            prefix = base_name.lower()

        # 确保前缀有效
        if not prefix:
            print(f"Warning: Could not derive a valid prefix for {file_name_full}. Skipping.")
            continue

        # 列数检查
        if df.shape[1] < 2:
            print(
                f"Warning: File {file_name_full} has fewer than 2 columns. Expected 'date' and a price column. Skipping this file.")
            continue

        # 重命名列：第一列为 'date'，第二列为 '{prefix}_price'
        original_columns = df.columns
        price_column_name = f'{prefix}_price'

        df = df.rename(columns={original_columns[0]: 'date', original_columns[1]: price_column_name})

        # 只选择 'date' 和新的价格列
        if price_column_name not in df.columns or 'date' not in df.columns:
            print(
                f"Warning: Expected columns 'date' or '{price_column_name}' not found after renaming in {file_name_full}. Original columns were {original_columns}. Skipping.")
            continue
        df = df[['date', price_column_name]]

        # 将 'date' 列转换为 datetime 对象
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            print(
                f"Warning: Could not convert 'date' column to datetime for {file_name_full}. Error: {e}. Trying to infer format or skipping.")
            try:
                df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
                print(f"Successfully converted 'date' with infer_datetime_format for {file_name_full}.")
            except Exception as e_infer:
                print(
                    f"Still could not convert 'date' column for {file_name_full} after inferring: {e_infer}. Skipping this file.")
                continue

        # 删除在 'date' 或价格列中包含任何缺失值 (NaN) 的行
        df_cleaned = df.dropna(subset=['date', price_column_name])

        if len(df_cleaned) < len(df):
            print(f"Dropped {len(df) - len(df_cleaned)} rows with missing values from {file_name_full}.")

        if df_cleaned.empty:
            print(
                f"Warning: File {file_name_full} is empty after cleaning (dropna). Original row count before dropna: {len(df)}. Skipping this file.")
            continue

        df = df_cleaned

        # 与主 DataFrame 合并
        if merged_df is None:
            merged_df = df
        else:
            if 'date' not in merged_df.columns:
                print(
                    f"Critical Error: 'date' column missing in master merged_df before merging {file_name_full}. Stopping.")
                break
            if 'date' not in df.columns:
                print(f"Critical Error: 'date' column missing in {file_name_full} before merging. Skipping this file.")
                continue

            merged_df = pd.merge(merged_df, df, on='date', how='inner')

        print(
            f"Successfully processed and merged {file_name_full}. Current merged shape: {merged_df.shape if merged_df is not None else 'None'}")

    if merged_df is not None and not merged_df.empty:
        # 按日期排序
        merged_df = merged_df.sort_values(by='date').reset_index(drop=True)

        # 保存最终合并的 DataFrame 为 XLSX 文件
        output_file_name = "price_data.xlsx"  # 输出文件名已更改
        merged_df.to_excel(output_file_name, index=False) # 输出为Excel，不包含索引
        print(f"\nSuccessfully merged all Excel files. Output saved to {output_file_name}")
        print("\nFinal merged data preview (first 5 rows):")
        print(merged_df.head())
        print("\nFinal merged data info:")
        merged_df.info()
    elif merged_df is not None and merged_df.empty:
        print(
            "\nProcessing complete, but the final merged DataFrame is empty. This might be due to no common dates after filtering, issues with input files, or all data being filtered out.")
    else:
        print(
            "\nNo data was processed or merged successfully. Please check the input files and console logs for errors or warnings.")

except Exception as e:
    print(f"An unexpected error occurred during the script execution: {e}")

print("\nData processing finished.")