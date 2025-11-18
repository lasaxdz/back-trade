# 使用TdxHq_API的示例
# 知乎：https://zhuanlan.zhihu.com/p/1951565268410139393
from pytdx.hq import TdxHq_API
from pytdx.util.best_ip import select_best_ip
import pandas as pd

def auto_get_best_ip_and_data():
    """自动选择最优服务器并获取股票数据的完整示例"""
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
            # 将股票池转换为pytdx需要的(market, code)格式
            stock_codes = []
            for stock in STOCK_POOL:
                code, suffix = stock.split('.')
                market = 1 if suffix == 'SH' else 0
                stock_codes.append((market, code))
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


# 股票池: 11.17
STOCK_POOL = [
    '000034.SZ', '000039.SZ', '000158.SZ', '000333.SZ', '000338.SZ',
    '000426.SZ', '000536.SZ', '000555.SZ', '000559.SZ', '000564.SZ',
    '000566.SZ', '000572.SZ', '000620.SZ', '000632.SZ', '000657.SZ',
    '000723.SZ', '000792.SZ', '000796.SZ', '000833.SZ', '000973.SZ',
    '001203.SZ', '001309.SZ', '002028.SZ', '002129.SZ', '002163.SZ',
    '002176.SZ', '002192.SZ', '002208.SZ', '002218.SZ', '002240.SZ',
    '002250.SZ', '002251.SZ', '002255.SZ', '002261.SZ', '002298.SZ',
    '002317.SZ', '002326.SZ', '002340.SZ', '002402.SZ', '002426.SZ',
    '002451.SZ', '002460.SZ', '002466.SZ', '002497.SZ', '002506.SZ',
    '002639.SZ', '002709.SZ', '002728.SZ', '002738.SZ', '002741.SZ',
    '002756.SZ', '002759.SZ', '002805.SZ', '002812.SZ', '002837.SZ',
    '600016.SH', '600036.SH', '600089.SH', '600096.SH', '600110.SH',
    '600141.SH', '600157.SH', '600203.SH', '600309.SH', '600338.SH',
    '600376.SH', '600408.SH', '600410.SH', '600418.SH', '600438.SH',
    '600519.SH', '600550.SH', '600556.SH', '600693.SH', '600711.SH',
    '600745.SH', '600759.SH', '600875.SH', '600967.SH', '601012.SH',
    '601166.SH', '601179.SH', '601288.SH', '601318.SH', '601360.SH',
    '601398.SH', '601600.SH', '601606.SH', '601888.SH', '601969.SH',
    '601988.SH', '603026.SH', '603067.SH', '603185.SH', '603260.SH',
    '603659.SH', '603686.SH', '603799.SH', '603881.SH'
]

if __name__ == "__main__":
    auto_get_best_ip_and_data()


