# 使用TdxHq_API的示例
# 知乎：https://zhuanlan.zhihu.com/p/1951565268410139393
from pytdx.hq import TdxHq_API
from pytdx.util.best_ip import select_best_ip
import pandas as pd
def auto_get_best_ip_and_data():
    """
    自动选择最优服务器并获取股票数据的完整示例
    """
    # 1. 自动选择最佳行情服务器
    print("正在自动测试最优服务器...")
    try:
        best_ip_info = select_best_ip()
        best_ip = best_ip_info['ip']
        best_port = best_ip_info['port']
        print(f"✅ 找到最优服务器: {best_ip}:{best_port}")
    except Exception as e:
        print(f"❌ 自动选择最优服务器时发生错误: {e}")
        return None
    # 2. 创建API对象
    api = TdxHq_API()
    # 3. 连接服务器并获取数据
    try:
        with api.connect(best_ip, best_port):  # 使用with语句确保连接被正确关闭
            print("✅ 服务器连接成功!")
            # 示例1: 获取多只股票的实时行情
            stock_codes = [(1, '600519'), (0, '000001')]  # 沪市贵州茅台和深市平安银行
            quotes = api.get_security_quotes(stock_codes)
            df_quotes = api.to_df(quotes)
            # 简单处理一下数据，只保留一些重要字段
            important_columns = ['code', 'name', 'price', 'last_close', 'open', 'high', 'low', 'vol', 'amount']
            # 确保这些字段在DataFrame中存在
            existing_columns = [col for col in important_columns if col in df_quotes.columns]
            df_quotes_display = df_quotes[existing_columns]
            # 计算涨跌幅
            if 'price' in df_quotes.columns and 'last_close' in df_quotes.columns:
                df_quotes_display['change_percent'] = (df_quotes['price'] / df_quotes['last_close'] - 1) * 100
                df_quotes_display['change_percent'] = df_quotes_display['change_percent'].round(2)
            print("\n📊 实时行情数据:")
            print(df_quotes_display)
            # 示例2: 获取贵州茅台的日K线数据
            k_lines = api.get_security_bars(9, 1, '600519', 0, 10)  # 9表示日K线，1表示沪市，获取10条数据
            df_k_lines = api.to_df(k_lines)
            if not df_k_lines.empty:
                # 选择需要显示的列
                k_line_columns = ['datetime', 'open', 'close', 'high', 'low', 'vol']
                existing_k_line_columns = [col for col in k_line_columns if col in df_k_lines.columns]
                df_k_lines_display = df_k_lines[existing_k_line_columns]
                print("\n📈 贵州茅台日K线数据:")
                print(df_k_lines_display)
            return df_quotes_display, df_k_lines_display
    except Exception as e:
        print(f"❌ 在连接服务器或获取数据时发生错误: {e}")
        return None, None
if __name__ == "__main__":
    auto_get_best_ip_and_data()



# 运行日志：
# (money) D:\Sync\Zaqi\3.my_trade>C:/Users/JXGM/miniconda3/envs/money/python.exe d:/Sync/Zaqi/3.my_trade/backtest_pytdx.py
# 正在自动测试最优服务器...
# BAD RESPONSE 106.120.74.86
# BAD RESPONSE 113.105.73.88
# BAD RESPONSE 113.105.73.88
# BAD RESPONSE 114.80.80.222
# BAD RESPONSE 117.184.140.156
# BAD RESPONSE 119.147.171.206
# BAD RESPONSE 119.147.171.206
# BAD RESPONSE 218.108.50.178
# BAD RESPONSE 221.194.181.176
# BAD RESPONSE 106.120.74.86
# BAD RESPONSE 112.95.140.74
# BAD RESPONSE 112.95.140.92
# BAD RESPONSE 112.95.140.93
# BAD RESPONSE 113.05.73.88
# BAD RESPONSE 114.67.61.70
# BAD RESPONSE 114.80.149.19
# BAD RESPONSE 114.80.149.22
# BAD RESPONSE 114.80.149.84
# BAD RESPONSE 114.80.80.222
# GOOD RESPONSE 115.238.56.198
# GOOD RESPONSE 115.238.90.165
# BAD RESPONSE 117.184.140.156
# BAD RESPONSE 119.147.164.60
# BAD RESPONSE 119.147.171.206
# BAD RESPONSE 119.29.51.30
# BAD RESPONSE 121.14.104.70
# BAD RESPONSE 121.14.104.72
# BAD RESPONSE 121.14.110.194
# BAD RESPONSE 121.14.2.7
# BAD RESPONSE 123.125.108.23
# BAD RESPONSE 123.125.108.24
# BAD RESPONSE 124.160.88.183
# BAD RESPONSE 180.153.18.17
# GOOD RESPONSE 180.153.18.170
# BAD RESPONSE 180.153.18.171
# BAD RESPONSE 180.153.39.51
# BAD RESPONSE 218.108.47.69
# BAD RESPONSE 218.108.50.178
# BAD RESPONSE 218.108.98.244
# GOOD RESPONSE 218.75.126.9
# BAD RESPONSE 218.9.148.108
# BAD RESPONSE 221.194.181.176
# BAD RESPONSE 59.173.18.69
# GOOD RESPONSE 60.12.136.250
# GOOD RESPONSE 60.191.117.167
# BAD RESPONSE 60.28.29.69
# BAD RESPONSE 61.135.142.73
# BAD RESPONSE 61.135.142.88
# BAD RESPONSE 61.152.107.168
# BAD RESPONSE 61.152.249.56
# BAD RESPONSE 61.153.144.179
# BAD RESPONSE 61.153.209.138
# BAD RESPONSE 61.153.209.139
# BAD RESPONSE hq.cjis.cn
# BAD RESPONSE hq1.daton.com.cn
# GOOD RESPONSE jstdx.gtjas.com
# GOOD RESPONSE shtdx.gtjas.com
# GOOD RESPONSE sztdx.gtjas.com
# BAD RESPONSE 113.105.142.162
# BAD RESPONSE 23.129.245.199
# ✅ 找到最优服务器: shtdx.gtjas.com:7709
# ✅ 服务器连接成功!
# d:\Sync\Zaqi\3.my_trade\backtest_pytdx.py:637: SettingWithCopyWarning: 
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   df_quotes_display['change_percent'] = (df_quotes['price'] / df_quotes['last_close'] - 1) * 100
# d:\Sync\Zaqi\3.my_trade\backtest_pytdx.py:638: SettingWithCopyWarning: 
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   df_quotes_display['change_percent'] = df_quotes_display['change_percent'].round(2)

# 📊 实时行情数据:
#      code    price  last_close     open     high      low     vol        amount  change_percent
# 0  600519  1471.00     1456.60  1454.00  1473.00  1445.79   34462  5.035780e+09            0.99
# 1  000001    11.67       11.75    11.75    11.75    11.62  995232  1.161416e+09           -0.68

# 📈 贵州茅台日K线数据:
#            datetime     open    close     high      low      vol
# 0  2025-11-04 15:00  1435.10  1429.00  1435.78  1423.78  26565.0
# 1  2025-11-05 15:00  1425.89  1420.08  1430.99  1420.01  34475.0
# 2  2025-11-06 15:00  1430.00  1435.13  1441.45  1429.99  38347.0
# 3  2025-11-07 15:00  1435.11  1433.33  1439.78  1431.11  18861.0
# 4  2025-11-10 15:00  1435.00  1462.30  1463.69  1434.98  49451.0
# 5  2025-11-11 15:00  1462.00  1458.99  1462.18  1447.00  26691.0
# 6  2025-11-12 15:00  1459.98  1465.15  1478.36  1459.21  32992.0
# 7  2025-11-13 15:00  1462.12  1470.38  1473.58  1458.00  31179.0
# 8  2025-11-14 15:00  1470.00  1456.60  1478.95  1456.30  27473.0
# 9  2025-11-17 15:00  1454.00  1471.00  1473.00  1445.79  34462.0

# (money) D:\Sync\Zaqi\3.my_trade>
