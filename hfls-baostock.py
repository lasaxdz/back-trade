import baostock as bs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import warnings
# 设置警告过滤和中文字体
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 辅助函数 ====================
def convert_symbol_to_baostock(symbol):
    """格式转换如600438.SS -> sh.600438, 002130.SZ -> sz.002130"""
    code = symbol.split('.')[0]
    if symbol.endswith('.SS'):
        return f'sh.{code}'
    elif symbol.endswith('.SZ'):
        return f'sz.{code}'
    return symbol

def get_stock_hist_data(symbol, start_date, end_date, adjustflag='2'):
    """使用Baostock获取单只股票历史数据;adjustflag: '1'后复权, '2'前复权, '3'不复权"""
    try:
        # 转换股票代码格式
        code = convert_symbol_to_baostock(symbol)
        # 获取股票日K线数据
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag
        )
        if rs.error_code != '0':
            print(f'获取 {symbol} 数据失败: {rs.error_msg}')
            return None
        # 转换为DataFrame
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        if not data_list:
            return None
        df = pd.DataFrame(data_list, columns=rs.fields)
        # 转换数据类型
        df['date'] = pd.to_datetime(df['date'])
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        # 设置索引
        df = df.set_index('date')
        # 重命名列以匹配原有格式
        df = df.rename(columns={
            'amount': 'turnover_money'  # 成交额
        })
        # 过滤停牌日期（tradestatus='0'表示停牌）
        df = df[df['tradestatus'] == '1']
        # 删除不需要的列
        df = df.drop(columns=['adjustflag', 'tradestatus', 'isST'], errors='ignore')
        return df
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None

def get_all_stock_data(symbol_list, start_date, end_date):
    """下载自定义股票池的股票数据"""
    all_data = {}
    failed_symbols = []
    print(f"正在下载 {len(symbol_list)} 只股票数据...")
    for i, symbol in enumerate(symbol_list):
        df = get_stock_hist_data(symbol, start_date, end_date)
        if df is not None and not df.empty and len(df) > 30:
            all_data[symbol] = df
        else:
            print(f"警告: {symbol} 数据不足或为空，跳过")
            failed_symbols.append(symbol)
        time.sleep(0.1)  # 避免请求过快
    print(f"成功下载 {len(all_data)} 只股票，失败 {len(failed_symbols)} 只")
    if failed_symbols:
        print(f"失败列表: {failed_symbols}")
    return all_data, failed_symbols

def prepare_indicators(all_data):
    """为所有股票数据添加技术指标"""
    for symbol, df in all_data.items():
        if 'turnover_money' not in df.columns:
            df['turnover_money'] = df['close'] * df['volume']
        df['ma5'] = df['close'].rolling(5).mean()

def get_index_data(start_date, end_date):
    """获取上证指数数据"""
    try:
        print("下载上证指数数据...")
        # 获取上证指数历史数据
        rs = bs.query_history_k_data_plus(
            "sh.000001",
            "date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # 指数不需要复权
        )
        if rs.error_code != '0':
            print(f'获取上证指数失败: {rs.error_msg}')
            return None
        # 转换为DataFrame
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        if not data_list:
            return None
        df = pd.DataFrame(data_list, columns=rs.fields)
        # 转换数据类型
        df['date'] = pd.to_datetime(df['date'])
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        # 设置索引
        df = df.set_index('date')
        print(f"成功获取上证指数数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"下载上证指数失败: {e}")
        return None

# ==================== 3. 选股、过滤与排名函数 ====================
def get_stock_list(all_data, current_date, sold_stock, portfolio_positions):
    """获取股票列表 - 每日更新股池"""
    stock_list = STOCK_POOL.copy()
    # 1. 保留原始股票池
    # 2. 保留可用股票
    # 3. 初级筛选：成交金额不低于10亿
    stock_list = filter_turnover_below_billion(stock_list, all_data, current_date)
    # 4. 初级排名：按当日成交金额/前一日成交金额从小到大排序
    stock_list = get_stock_rank_turnover_ratio(all_data, stock_list, current_date)
    # 5. 高级过滤：
    # a. 过滤价格变动超出9%的股票
    stock_list = filter_price_change(stock_list, all_data, current_date)
    # b. 过滤低于5日均线的个股
    stock_list = filter_price_above_ma5(stock_list, all_data, current_date)
    # c. 过滤冷却期内的股票
    stock_list = filter_buyagain(stock_list, sold_stock)
    # 输出终极排名信息
    print(f"[{current_date.strftime('%Y-%m-%d')}] 终极排名结果:")
    print(f"- 符合条件的股票总数: {len(stock_list)}")
    if len(stock_list) > 0:
        print(f"- 排名前10的股票: {', '.join(stock_list[:10])}")
    return stock_list

def get_stock_rank_turnover_ratio(all_data, stock_list, current_date):
    """股票排名 - 按当日成交金额/前一日成交金额从小到大排序"""
    if not stock_list:
        return []
    ranked_stocks = []
    turnover_ratios = []
    for symbol in stock_list:
        if symbol not in all_data:
            continue
        df = all_data[symbol]
        if current_date not in df.index or len(df) < 2:
            continue
        current_idx = df.index.get_loc(current_date)
        if current_idx < 1:
            continue
        if 'turnover_money' in df.columns:
            current_turnover = df.loc[current_date, 'turnover_money']
            prev_turnover = df.iloc[current_idx-1]['turnover_money']
        else:
            current_turnover = df.loc[current_date, 'close'] * df.loc[current_date, 'volume']
            prev_turnover = df.iloc[current_idx-1]['close'] * df.iloc[current_idx-1]['volume']
        # 避免除以零
        if prev_turnover > 0:
            turnover_ratio = current_turnover / prev_turnover
            turnover_ratios.append(turnover_ratio)
            ranked_stocks.append(symbol)
    if not ranked_stocks:
        return []
    scores = [(ranked_stocks[i], turnover_ratios[i]) for i in range(len(ranked_stocks))]
    # 从小到大排序
    scores.sort(key=lambda x: x[1])
    return [score[0] for score in scores[:min(100, len(scores))]]

def filter_price_above_ma5(stock_list, all_data, current_date):
    """过滤股价小于5日均线的个股"""
    filtered = []
    for stock in stock_list:
        if stock not in all_data:
            continue
        df = all_data[stock]
        if current_date not in df.index:
            continue
        current_close = df.loc[current_date, 'close']
        ma5 = df.loc[current_date, 'ma5'] if 'ma5' in df.columns else None
        if pd.isna(ma5):
            continue
        if current_close >= ma5:
            filtered.append(stock)
    return filtered

def filter_turnover_below_billion(stock_list, all_data, current_date):
    """过滤当日成交金额低于10亿的股票"""
    filtered = []
    for stock in stock_list:
        if stock not in all_data:
            continue
        df = all_data[stock]
        if current_date not in df.index:
            continue
        # 当日成交金额
        if 'turnover_money' in df.columns:
            current_turnover = df.loc[current_date, 'turnover_money']
        else:
            current_price = df.loc[current_date, 'close']
            current_volume = df.loc[current_date, 'volume']
            current_turnover = current_price * current_volume
        if current_turnover >= params.min_turnover:
            filtered.append(stock)
    return filtered

def filter_price_change(stock_list, all_data, current_date, max_gain=9, min_gain=-9):
    """过滤价格变动超出指定范围的股票"""
    filtered = []
    for stock in stock_list:
        if stock not in all_data:
            continue
        df = all_data[stock]
        if current_date not in df.index or len(df) < 2:
            continue
        current_idx = df.index.get_loc(current_date)
        if current_idx < 1:
            continue
        current_close = df.loc[current_date, 'close']
        prev_close = df.iloc[current_idx-1]['close']
        gain = (current_close - prev_close) / prev_close * 100
        if min_gain < gain < max_gain:
            filtered.append(stock)
    return filtered

def filter_buyagain(stock_list, sold_stock):
    """过滤卖出不足params.buyagain日的股票"""
    return [stock for stock in stock_list if stock not in sold_stock]

# ==================== 4. 交割单函数 ====================
def print_trade_settlement(trade_log, portfolio, cash, last_date, all_data):
    """打印交易交割单"""
    if not trade_log:
        print("\n没有交易记录")
        return
    print("\n" + "=" * 80)
    print("交易交割单 (模拟)")
    print("=" * 80)
    headers = ["交易日期", "股票代码", "操作", "价格(元)", "数量(股)"]
    print("{:<9} {:<9} {:<9} {:<9} {:<9}".format(*headers))
    print("-" * 80)
    for trade in trade_log:
        date_str = trade['date'].strftime('%Y-%m-%d')
        symbol = trade['symbol']
        action = trade['action']
        price = trade['price']
        shares = trade['shares']
        print(f"{date_str:<12} {symbol:<12} {action:<12} {price:<12} {shares:<12}")
    print("-" * 80)

# ==================== 4. 回测主循环 ====================
def run_backtest():
    """运行回测"""
    # 登录Baostock系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f'登录失败: {lg.error_msg}')
        return None, None
    print('登录Baostock成功')
    # 用于控制是否输出每日排名（避免输出过多）
    print_rank_interval = 5  # 每5个交易日输出一次
    try:
        print("=" * 60)
        print(f"单因子选股策略回测 (Baostock数据源)")
        print(f"回测期间: {START_DATE} 至 {END_DATE}")
        print(f"初始资金: {params.initial_capital:,.2f} 元")
        print("=" * 60)
        all_data, failed_symbols = get_all_stock_data(STOCK_POOL, START_DATE, END_DATE)
        prepare_indicators(all_data)
        # 2. 下载上证指数
        index_df = get_index_data(START_DATE, END_DATE)
        if index_df is None or index_df.empty:
            print("上证指数数据获取失败，回测终止")
            return None, None
        all_data['000001.SS'] = index_df
        if len(all_data) < 2:
            print("数据不足，无法回测")
            return None, None
        # 3. 初始化
        cash = params.initial_capital
        portfolio = {}
        sold_stock = {}
        trade_log = []
        equity_curve = []
        dates = index_df.index
        # 4. 回测主循环
        print("开始回测...")
        total_days = len(dates)
        for i, current_date in enumerate(dates):
            progress_step = max(1, total_days // 10)
            if (i + 1) % progress_step == 0 or i == total_days - 1:
                progress = (i + 1) / total_days * 100
                print(f"回测进度: {progress:.1f}% ({i+1}/{total_days})")
            # 更新卖出计数器
            to_remove = []
            for stock in sold_stock:
                sold_stock[stock] -= 1
                if sold_stock[stock] <= 0:
                    to_remove.append(stock)
            for stock in to_remove:
                del sold_stock[stock]
            # 每日选股
            candidate_stocks = get_stock_list(all_data, current_date, sold_stock, portfolio.keys())
            # 1. 卖出规则检查：
            # a. 止损卖出：股价跌破5日均线时卖出
            for stock in list(portfolio.keys()):
                df = all_data.get(stock)
                if df is None or current_date not in df.index:
                    continue
                current_price = df.loc[current_date, 'close']
                ma5 = df.loc[current_date, 'ma5'] if 'ma5' in df.columns else None
                if pd.isna(ma5):
                    continue
                if current_price < ma5:
                    price = current_price
                    shares = portfolio[stock]
                    if shares > 0:
                        sell_reason = '止损'
                        amount = shares * price
                        commission = max(amount * params.commission_rate, params.min_commission)
                        net_amount = amount - commission
                        trade_log.append({
                            'date': current_date,
                            'symbol': stock,
                            'action': sell_reason,
                            'price': price,
                            'shares': shares,
                            'amount': amount,
                            'commission': commission
                        })
                        cash += net_amount
                        portfolio[stock] = 0
                        sold_stock[stock] = params.buyagain
            # 2. 买入规则：买入终极排名中排序靠前的股票，单一股票等权重配置
            if candidate_stocks:
                total_value = cash + sum(portfolio[s] * (all_data[s].loc[current_date, 'close'] if s in all_data and current_date in all_data[s].index else 0) for s in portfolio)
                # 确定要购买的股票数量
                target_stocknum = min(params.stocknum, len(candidate_stocks))
                if target_stocknum > 0:
                    buycash_per_stock = total_value / target_stocknum
                else:
                    buycash_per_stock = 0
                # 买入排名靠前的股票
                buy_list = candidate_stocks[:target_stocknum]
                for stock in buy_list:
                    if stock not in all_data or current_date not in all_data[stock].index:
                        continue
                    price = all_data[stock].loc[current_date, 'close']
                    if price <= 0:
                        continue
                    current_shares = portfolio.get(stock, 0)
                    current_value = current_shares * price
                    # 等权重配置：单一股票配置相同资金
                    target_value = buycash_per_stock
                    # 如果当前持仓已经接近目标值，跳过
                    if current_value >= target_value * 0.9:
                        continue
                    tobuy_value = target_value - current_value
                    if tobuy_value <= 0:
                        continue
                    # 计算可买入股数（取整百）
                    shares_to_buy = int(tobuy_value / (price * (1 + params.commission_rate)))
                    shares_to_buy = max(100, (shares_to_buy // 100) * 100)
                    if shares_to_buy <= 0:
                        continue
                    # 计算实际成本
                    buy_amount = shares_to_buy * price
                    commission = max(buy_amount * params.commission_rate, params.min_commission)
                    total_cost = buy_amount + commission
                    # 确保有足够现金
                    if total_cost > cash * 0.99:
                        shares_to_buy = int(cash * 0.99 / (price * (1 + params.commission_rate)))
                        shares_to_buy = max(100, (shares_to_buy // 100) * 100)
                        if shares_to_buy <= 0:
                            continue
                        buy_amount = shares_to_buy * price
                        commission = max(buy_amount * params.commission_rate, params.min_commission)
                        total_cost = buy_amount + commission
                    # 执行买入
                    cash -= total_cost
                    portfolio[stock] = portfolio.get(stock, 0) + shares_to_buy
                    trade_log.append({
                        'date': current_date,
                        'symbol': stock,
                        'action': '买入',
                        'price': price,
                        'shares': shares_to_buy,
                        'amount': buy_amount,
                        'commission': commission
                    })
            # 记录净值
            total_value = cash + sum(portfolio[s] * (all_data[s].loc[current_date, 'close'] if s in all_data and current_date in all_data[s].index else 0) for s in portfolio)
            equity_curve.append(total_value)
        # 5. 输出结果
        final_value = equity_curve[-1]
        total_return = (final_value / params.initial_capital - 1) * 100
        # 计算基准收益
        index_return = 0.0
        if '000001.SS' in all_data and not all_data['000001.SS'].empty:
            index_df = all_data['000001.SS']
            first_date = index_df.index[0]
            last_date = index_df.index[-1]
            if 'close' in index_df.columns:
                start_index = index_df.loc[first_date, 'close']
                end_index = index_df.loc[last_date, 'close']
                index_return = (end_index / start_index - 1) * 100
        print_trade_settlement(trade_log, portfolio, cash, dates[-1], all_data)
        print(f"\n=== 回测结果 ===")
        print(f"起始资金: {params.initial_capital:,.2f} 元")
        print(f"结束资金: {final_value:,.2f} 元")
        print(f"总收益率: {total_return:.2f}%")
        print(f"上证指数基准收益率: {index_return:.2f}%")
        plot_results(all_data, equity_curve, dates, trade_log, params.initial_capital)
        return equity_curve, trade_log
    finally:
        # 登出Baostock系统
        bs.logout()
        print('登出Baostock系统')

# ==================== 4. 可视化 ====================
def plot_results(all_data, equity_curve, dates, trade_log, initial_capital):
    """绘制回测结果"""
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    index_df = all_data['000001.SS']
    ax1.plot(index_df.index, index_df['close'], label='上证指数', color='black')
    ma10 = index_df['close'].rolling(10).mean()
    ax1.plot(ma10.index, ma10, label='10日均线', color='orange', linestyle='--')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(dates, equity_curve, label='策略净值', color='blue', linewidth=2)
    index_start = index_df['close'].iloc[0]
    index_normalized = [initial_capital * (price / index_start) for price in index_df['close']]
    ax2.plot(index_df.index, index_normalized, label='上证指数基准', color='red', alpha=0.7)
    ax1.set_title('上证指数 & 10日均线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_title('净值曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
        ax.xaxis.set_major_locator(locator)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# ==================== 8. 主程序 ====================
def load_stock_pool_from_file(path):
    """从文件加载股票池"""
    syms = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('STOCK_POOL'):
                    continue
                token = line.split()[0]
                if len(token) == 6 and token.isdigit():
                    if token.startswith(('6', '688', '605', '603', '601')):
                        syms.append(f"{token}.SS")
                    else:
                        syms.append(f"{token}.SZ")
                elif token.endswith('.SS') or token.endswith('.SZ'):
                    syms.append(token)
        if syms:
            return syms
    except Exception:
        pass
    return [
        "601360.SS", "601012.SS", "600089.SS", "601888.SS", "603799.SS",
        "601899.SS", "601318.SS", "600111.SS", "600438.SS",
        "002130.SZ", "002466.SZ", "002460.SZ", "000002.SZ", "000858.SZ",
        "000063.SZ", "002384.SZ", "000568.SZ", "000792.SZ"
    ]

class StrategyParams:
    def __init__(self):
        self.stocknum = 4
        self.buyagain = 5
        self.commission_rate = 0.0001
        self.min_commission = 0.01
        self.initial_capital = 1_0000.0
        self.min_turnover = 10_0000_0000
params = StrategyParams()

START_DATE = '2025-09-01'
END_DATE = '2025-11-01'
STOCK_POOL = load_stock_pool_from_file(r'd:\Sync\Zaqi\3.my_trade\主板非ST.txt')

if __name__ == '__main__':
    result = run_backtest()
    if result is not None:
        equity_curve, trade_log = result
        print("\n✓ 回测完成")
    else:
        print("\n✗ 回测失败")
