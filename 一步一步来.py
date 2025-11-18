# 使用TdxHq_API的示例
# 知乎：https://zhuanlan.zhihu.com/p/1951565268410139393
from pytdx.hq import TdxHq_API
from pytdx.util.best_ip import select_best_ip
import pandas as pd
import json
import time
import os

def load_cached_server():
    """加载缓存的最优服务器信息"""
    cache_file = "data_cache/server_cache.json"
    cache_expire = 1 * 24 * 3600  # 缓存有效期1天
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            # 检查缓存是否过期
            if time.time() - cache_data.get("timestamp", 0) < cache_expire:
                return cache_data.get("server", None)
        except Exception as e:
            print(f"❌ 加载缓存时发生错误: {e}")
    return None

def save_cached_server(server_info):
    """保存最优服务器信息到缓存文件"""
    cache_file = "data_cache/server_cache.json"
    cache_data = {
        "timestamp": time.time(),
        "server": server_info
    }
    try:
        # 确保缓存目录存在
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        print("✅ 服务器信息已缓存到本地")
    except Exception as e:
        print(f"❌ 保存缓存时发生错误: {e}")

def auto_get_best_ip_and_data():
    """自动选择最优服务器并获取股票数据的完整示例"""
    # 1. 首先尝试加载缓存的服务器信息
    cached_server = load_cached_server()
    if cached_server:
        best_ip = cached_server['ip']
        best_port = cached_server['port']
        print(f"✅ 使用缓存的最优服务器: {best_ip}:{best_port}")
    else:
        # 2. 如果缓存不存在或过期，重新选择最优服务器
        print("正在自动测试最优服务器...")
        try:
            best_ip_info = select_best_ip()
            best_ip = best_ip_info['ip']
            best_port = best_ip_info['port']
            print(f"✅ 找到最优服务器: {best_ip}:{best_port}")
            # 保存到缓存
            save_cached_server(best_ip_info)
        except Exception as e:
            print(f"❌ 自动选择最优服务器时发生错误: {e}")
            return None
    # 2. 创建API对象
    api = TdxHq_API()
    # 3. 连接服务器并获取数据
    try:
        with api.connect(best_ip, best_port):  # 使用with语句确保连接被正确关闭
            print("✅ 服务器连接成功!")
            # ============================
            # ============================
            # 将股票池转换为pytdx需要的(market, code)格式
            stock_codes = []
            for stock in STOCK_POOL:
                code, suffix = stock.split('.')
                market = 1 if suffix == 'SH' else 0
                stock_codes.append((market, code))
            # 将股票列表分成多个批次查询（pytdx可能有单次查询限制）
            batch_size = 50
            all_quotes = []
            for i in range(0, len(stock_codes), batch_size):
                batch = stock_codes[i:i+batch_size]
                batch_quotes = api.get_security_quotes(batch)
                all_quotes.extend(batch_quotes)
            quotes = all_quotes
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
            # ============================
            # ============================
            # 获取选股池中所有个股的日K线数据
            all_k_lines = []
            for stock in STOCK_POOL:
                # 解析股票代码和市场
                if stock.endswith('.SH'):
                    market = 1  # 沪市
                    code = stock[:-3]  # 去掉.SH后缀
                elif stock.endswith('.SZ'):
                    market = 0  # 深市
                    code = stock[:-3]  # 去掉.SZ后缀
                else:
                    continue  # 跳过不符合格式的股票代码
                # 获取日K线数据（9表示日K线，获取10条最新数据）
                k_lines = api.get_security_bars(9, market, code, 0, 10)
                df_k_line = api.to_df(k_lines)
                if not df_k_line.empty:
                    # 添加股票代码列
                    df_k_line['stock_code'] = stock
                    # 选择需要显示的列，添加amount（成交金额）字段
                    k_line_columns = ['stock_code', 'datetime', 'open', 'close', 'high', 'low', 'vol', 'amount']
                    existing_k_line_columns = [col for col in k_line_columns if col in df_k_line.columns]
                    df_k_line_display = df_k_line[existing_k_line_columns]
                    # 打印每只股票的K线数据
                    print(f"\n📈 {stock} 日K线数据:")
                    print(df_k_line_display)
                    all_k_lines.append(df_k_line_display)
            # 合并所有股票的K线数据
            if all_k_lines:
                df_k_lines_display = pd.concat(all_k_lines, ignore_index=True)
                # 按交易日分组，每组内按成交金额降序排列
                print(f"\n{'='*60}")
                print("📊 所有个股日K线数据按交易日汇总（按成交金额降序）")
                print(f"{'='*60}")
                # 遍历每个交易日，只打印第一日和最后一日
                groups = list(df_k_lines_display.groupby('datetime'))
                if groups:
                    # 打印第一日数据
                    first_datetime, first_data = groups[0]
                    sorted_first_data = first_data.sort_values(by='amount', ascending=False)
                    print(f"\n日期: {first_datetime} (第一日)")
                    print(sorted_first_data)
                    # 打印最后一日数据（避免与第一日重复）
                    last_datetime, last_data = groups[-1]
                    if first_datetime != last_datetime:
                        sorted_last_data = last_data.sort_values(by='amount', ascending=False)
                        print(f"\n日期: {last_datetime} (最后一日)")
                        print(sorted_last_data)
            else:
                df_k_lines_display = pd.DataFrame()  # 返回空DataFrame
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
    '603659.SH', '603686.SH', '603799.SH', '603881.SH', '603993.SH',
]

if __name__ == "__main__":
    auto_get_best_ip_and_data()



