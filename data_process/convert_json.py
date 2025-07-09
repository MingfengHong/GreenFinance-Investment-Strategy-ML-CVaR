import json
import pandas as pd


def json_to_table(file_path="CDCGBI.json"):
    """
    从指定的 JSON 文件中提取数据并转换为 pandas DataFrame，
    同时将 tradingDate 列的格式从 YYYYMMDD 转换为 YYYY-MM-DD。

    参数:
    file_path (str): JSON 文件的路径。

    返回:
    pandas.DataFrame: 包含提取和转换后数据的 DataFrame，如果出错则返回 None。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)

        data_list = content.get("data")

        if data_list is None:
            print(f"错误: 文件 '{file_path}' 中未找到 'data' 键。")
            return None

        if not isinstance(data_list, list):
            print(f"错误: 文件 '{file_path}' 中 'data' 键对应的值不是列表。")
            return None

        df = pd.DataFrame(data_list)

        # 检查 'tradingDate' 列是否存在
        if 'tradingDate' in df.columns:
            try:
                # 将 'tradingDate' 从 YYYYMMDD 字符串转换为 datetime 对象
                df['tradingDate'] = pd.to_datetime(df['tradingDate'], format='%Y%m%d')
                # 将 datetime 对象格式化为 YYYY-MM-DD 字符串
                df['tradingDate'] = df['tradingDate'].dt.strftime('%Y-%m-%d')
            except ValueError as ve:
                print(f"警告: 'tradingDate' 列包含无效的日期格式，无法转换: {ve}")
            except Exception as e:
                print(f"警告: 转换 'tradingDate' 列时发生错误: {e}")
        else:
            print("警告: DataFrame 中未找到 'tradingDate' 列。")

        return df

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的 JSON 格式。")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None


# 主程序执行部分
if __name__ == "__main__":
    file_name = "CDCGBI.json"

    data_table = json_to_table(file_name)

    if data_table is not None:
        print("提取并转换格式后的数据表格：")
        print(data_table)

        try:
            csv_file_name = "CDCGBI_table_formatted_date.csv"
            data_table.to_csv(csv_file_name, index=False, encoding='utf-8-sig')
            print(f"\n数据已成功保存到 '{csv_file_name}'")
        except Exception as e:
            print(f"\n保存到 CSV 文件时出错: {e}")