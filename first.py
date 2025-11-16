# -*- coding: utf-8 -*-
"""
数据源: Baostock (http://baostock.com/)
单因子选股策略 → Baostock + pandas 回测 + 可视化 + 交割单
依赖安装：pip install pandas numpy matplotlib baostock
===========
1, 所有的操作, 都是按“收盘价”操作的;
2, BaoStock.com 当前交易日18:00, 完成复权因子数据入库;
"""

# ==================== 1. 基础库 ====================
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

# ==================== 2. 辅助函数 ====================
def convert_symbol_to_baostock(symbol):
    """将Yahoo Finance格式的股票代码转换为Baostock格式。例如: 600438.SS -> sh.600438, 002130.SZ -> sz.002130"""
    code = symbol.split('.')[0]
    if symbol.endswith('.SS'):
        return f'sh.{code}'
    elif symbol.endswith('.SZ'):
        return f'sz.{code}'
    return symbol

def get_stock_hist_data(symbol, start_date, end_date, adjustflag='2'):
    """使用Baostock获取单只股票历史数据。adjustflag: '1'后复权, '2'前复权, '3'不复权"""
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

def get_all_stock_data(symbol_list, start_date, end_date, use_new_data=True):
    """下载自定义股票池的股票数据，支持数据缓存"""
    import os
    import pickle

    # 缓存文件路径
    cache_dir = 'stock_data_cache'
    os.makedirs(cache_dir, exist_ok=True)
    # 生成缓存文件名，包含日期范围信息
    cache_filename = f"{cache_dir}/stock_data_{start_date}_{end_date}.pkl"
    
    # 如果不使用新数据且缓存文件存在，则尝试加载缓存
    if not use_new_data and os.path.exists(cache_filename):
        try:
            print(f"正在从缓存文件加载数据: {cache_filename}")
            with open(cache_filename, 'rb') as f:
                cached_data = pickle.load(f)
            
            # 检查缓存数据是否包含所有需要的股票
            missing_symbols = [s for s in symbol_list if s not in cached_data['all_data']]
            
            if not missing_symbols:
                print(f"成功从缓存加载 {len(cached_data['all_data'])} 只股票数据")
                return cached_data['all_data'], cached_data['failed_symbols']
            else:
                print(f"缓存数据不完整，缺少 {len(missing_symbols)} 只股票")
        except Exception as e:
            print(f"加载缓存数据失败: {e}")
    
    # 下载新数据
    all_data = {}
    failed_symbols = []
    
    print(f"正在下载 {len(symbol_list)} 只股票数据...")
    
    for i, symbol in enumerate(symbol_list):
        df = get_stock_hist_data(symbol, start_date, end_date)
        if df is not None and not df.empty and len(df) >= 30:
            all_data[symbol] = df
        else:
            print(f"警告: {symbol} 数据不足或为空，跳过")
            failed_symbols.append(symbol)
        
        time.sleep(0.1)  # 避免请求过快
    
    print(f"成功下载 {len(all_data)} 只股票，失败 {len(failed_symbols)} 只")
    if failed_symbols:
        print(f"失败列表: {failed_symbols}")
    
    # 保存数据到缓存文件
    try:
        cache_data = {
            'all_data': all_data,
            'failed_symbols': failed_symbols,
            'timestamp': pd.Timestamp.now()
        }
        with open(cache_filename, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"数据已保存到缓存文件: {cache_filename}")
    except Exception as e:
        print(f"保存缓存数据失败: {e}")
    
    return all_data, failed_symbols


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


# ==================== 3. 过滤函数 ====================
def filter_paused_stock(stock_list, all_data, current_date):
    """优化停牌过滤函数（用成交量判断）"""
    return [
        stock for stock in stock_list 
        if (stock in all_data and 
            current_date in all_data[stock].index and 
            all_data[stock].loc[current_date, 'volume'] > 0)
    ]

def filter_buyagain(stock_list, sold_stock):
    """过滤卖出不足params.buyagain日的股票"""
    return [stock for stock in stock_list if stock not in sold_stock]

def filter_price_above_ma(stock_list, all_data, current_date, period=5):
    """通用均线过滤函数，替代单独的5日"""
    filtered = []
    for stock in stock_list:
        if stock not in all_data:
            continue
        
        df = all_data[stock]
        if current_date not in df.index or len(df) < period:
            continue
        
        current_close = df.loc[current_date, 'close']
        current_idx = df.index.get_loc(current_date)
        
        if current_idx < period - 1:
            continue
            
        # 计算period日均线
        ma_data = df.iloc[current_idx-period+1:current_idx+1]
        ma_value = ma_data['close'].mean()
        
        if current_close >= ma_value:
            filtered.append(stock)
    
    return filtered



def get_ma5(symbol, all_data, current_date):
    """获取5日均线值"""
    if symbol not in all_data:
        return 0
    
    df = all_data[symbol]
    if current_date not in df.index or len(df) < 5:
        return 0
    
    current_idx = df.index.get_loc(current_date)
    if current_idx < 4:
        return 0
    
    ma5 = df.iloc[current_idx-4:current_idx+1]['close'].mean()
    return ma5

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
        
        # 成交金额不低于10亿（1000000000元）
        if current_turnover >= 1000000000:
            filtered.append(stock)
    
    return filtered


def filter_high_daily_gain(stock_list, all_data, current_date):
    """过滤当日涨幅不低于9%的个股"""
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
        
        # 计算涨幅
        gain = (current_close - prev_close) / prev_close * 100
        
        # 过滤涨幅不低于9%的个股
        if gain <= 9:
            filtered.append(stock)
    
    return filtered

# ==================== 4. 选股与排名函数 ====================
def get_stock_list(all_data, current_date, sold_stock, portfolio_positions): # 这个参数后面会用到
    """获取股票列表 - 每日更新股池"""
    stock_list = STOCK_POOL.copy()
    
    # 2. 过滤停牌股票
    stock_list = filter_paused_stock(stock_list, all_data, current_date)
    # 3. 初级筛选：成交金额不低于10亿
    stock_list = filter_turnover_below_billion(stock_list, all_data, current_date)
    # 4. 初级排名：按当日成交金额/前一日成交金额从小到大排序
    stock_list = get_stock_rank_turnover_ratio(all_data, stock_list, current_date)
    
    # 5. 高级过滤：
    # a. 过滤当日涨幅不低于9%的个股
    stock_list = filter_high_daily_gain(stock_list, all_data, current_date)
    # b. 过滤低于5日均线的个股
    stock_list = filter_price_above_ma(stock_list, all_data, current_date, period=5)
    
    # 6. 过滤冷却期内的股票
    stock_list = filter_buyagain(stock_list, sold_stock)
    
    # 输出终极排名信息
    print(f"[{current_date.strftime('%Y-%m-%d')}] 终极排名结果:")
    print(f"- 符合条件的股票总数: {len(stock_list)}")
    if len(stock_list) > 0:
        print(f"- 排名前5的股票: {', '.join(stock_list[:5])}")
    
    return stock_list


def get_stock_rank_turnover_ratio(all_data, stock_list, current_date):
    """股票排名 - 按当日成交金额/前一日成交金额从小到大排序"""
    if not stock_list:
        return []
    
    scores = []
    for symbol in stock_list:
        if symbol not in all_data or current_date not in all_data[symbol].index:
            continue
        
        df = all_data[symbol]
        current_idx = df.index.get_loc(current_date)
        if current_idx < 1:
            continue
        
        # 获取成交金额（当日成交金额/前一日成交金额）
        if 'turnover_money' in df.columns:
            current_turnover = df.loc[current_date, 'turnover_money']
            prev_turnover = df.iloc[current_idx-1]['turnover_money']
        else:
            current_price, current_volume = df.loc[current_date, ['close', 'volume']]
            prev_data = df.iloc[current_idx-1]
            prev_price, prev_volume = prev_data['close'], prev_data['volume']
            current_turnover = current_price * current_volume
            prev_turnover = prev_price * prev_volume

        # 避免除以零
        if prev_turnover > 0:
            turnover_ratio = current_turnover / prev_turnover
            scores.append((symbol, turnover_ratio))
    
    # 从小到大排序
    scores.sort(key=lambda x: x[1])
    return [score[0] for score in scores[:min(100, len(scores))]]


# ==================== 5. 交割单函数 ====================
def print_trade_settlement(trade_log, portfolio, cash, last_date, all_data):
    """打印交易交割单"""
    if not trade_log:
        print("\n没有交易记录")
        return
    
    print("\n" + "=" * 80)
    print("交易交割单 (模拟)")
    print("=" * 80)
    
    headers = ["交易日期", "股票代码", "操作", "价格(元)", "数量(股)", "股票名称"]
    print("{:<12} {:<12} {:<12} {:<10} {:<12} {:<12}".format(*headers))
    print("-" * 95)
    
    for trade in trade_log:
        date_str = trade['date'].strftime('%Y-%m-%d')
        symbol = trade['symbol']
        name = CODE_NAME_MAP.get(symbol, '')
        action = trade['action']
        price = trade['price']
        shares = trade['shares']
        print(f"{date_str:<12} {symbol:<12} {action:<12} {price:<10.2f} {shares:<12.2f} {name:<12}")
    print("-" * 95)

# ==================== 6. 回测主循环 ====================
def run_backtest():
    """运行回测"""
    # 登录Baostock系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f'登录失败: {lg.error_msg}')
        return None, None
    print('登录Baostock成功')
    
    excel_stock_pool = load_stock_pool_from_excel(EXCEL_STOCK_POOL_PATH)
    if not excel_stock_pool:
        print("Excel股票池为空或读取失败，回测终止")
        bs.logout()
        return None, None
    else:
        global STOCK_POOL
        STOCK_POOL = excel_stock_pool
        print(f"STOCK_POOL 已用Excel更新，股票数: {len(STOCK_POOL)}")
        code_name_map = load_code_name_map_from_excel(EXCEL_STOCK_POOL_PATH)
        global CODE_NAME_MAP
        CODE_NAME_MAP = code_name_map
    
    try:
        print("=" * 60)
        print(f"单因子选股策略回测 (Baostock数据源)")
        print(f"回测期间: {START_DATE} 至 {END_DATE}")
        print(f"初始资金: {params.initial_capital:,.2f} 元")
        print("=" * 60)
        
        # 1. 下载股票数据（使用Excel更新后的 STOCK_POOL）
        all_data, failed_symbols = get_all_stock_data(STOCK_POOL, START_DATE, END_DATE, use_new_data=USE_NEW_DATA)
        
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
        portfolio = {}  # 持有股数
        stock_avg_cost = {}  # 平均买入成本
        sold_stock = {}
        trade_log = []
        equity_curve = []
        dates = index_df.index
        
        # 4. 回测主循环
        print("开始回测...")
        total_days = len(dates)
        for i, current_date in enumerate(dates):
            if (i + 1) % (total_days // 10) == 0 or i == total_days - 1:
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
                # 计算5日均线
                df = all_data.get(stock)
                if df is None or current_date not in df.index or len(df) < 5:
                    continue
                
                current_idx = df.index.get_loc(current_date)
                if current_idx < 4:
                    continue
                
                current_price = df.loc[current_date, 'close']
                ma5_data = df.iloc[current_idx-4:current_idx+1]
                ma5 = ma5_data['close'].mean()
                
                if current_price < ma5:
                    price = current_price
                    shares = portfolio[stock]
                    if shares > 0:
                        # 计算盈亏情况，判断是止损还是止盈
                        if stock in stock_avg_cost and stock_avg_cost[stock] > 0:
                            if price > stock_avg_cost[stock]:
                                sell_reason = '破线止盈'
                            else:
                                sell_reason = '破线止损'
                        else:
                            sell_reason = '破线卖出'  # 无法计算成本时的默认值
                        
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
            
            # 只保留破线卖出（止损）功能，移除调仓卖出
            
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
                    
                    # 更新持仓和平均成本
                    old_shares = portfolio.get(stock, 0)
                    old_cost = stock_avg_cost.get(stock, 0) * old_shares
                    new_shares = old_shares + shares_to_buy
                    new_cost = old_cost + buy_amount + commission
                    
                    portfolio[stock] = new_shares
                    stock_avg_cost[stock] = new_cost / new_shares if new_shares > 0 else 0
                    
                    trade_log.append({
                        'date': current_date,
                        'symbol': stock,
                        'action': '符合买入',
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
        
    # 登出Baostock系统
    finally:  
        bs.logout()

# ==================== 7. 可视化 ====================
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

def load_stock_pool_from_excel(excel_path, sheet_name='选股结果', max_count=100):
    """从Excel文件中读取股票池，最多max_count只股票（max_count会根据实际数量截取）"""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"读取股票池Excel失败: {e}")
        return []
    
    candidates = ['股票代码', '代码', 'symbol', 'Symbol', '证券代码']
    code_col = next((c for c in candidates if c in df.columns), None)
    if code_col is None:
        print("Excel表中未找到“股票代码”列，无法载入股票池")
        return []
    
    raw_series = df[code_col]
    raw_count = raw_series.notna().sum()

    codes = raw_series.dropna().astype(str).str.strip()
    # 只保留合理的代码格式：6位数字，可带 .SZ/.SH 后缀
    codes = codes[codes.str.match(r'^\d{6}(\.(SZ|SH))?$', na=False)]
    
    normalized = []
    for code in codes:
        if '.' in code:
            normalized.append(code.upper())
        else:
            clean = code.zfill(6)
            if clean.startswith('60'):
                normalized.append(f"{clean}.SH")
            else:
                normalized.append(f"{clean}.SZ")
    
    # 去重并按 max_count 截取
    dedup_ordered = list(dict.fromkeys(normalized))
    final_codes = dedup_ordered[:max_count] if max_count else dedup_ordered

    print(f"已从Excel载入股票池 原始{raw_count}，去重{len(dedup_ordered)}，最终{len(final_codes)} 只股票")
    return final_codes
    return normalized


def load_code_name_map_from_excel(excel_path, sheet_name='选股结果'):
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception:
        return {}
    candidates_code = ['股票代码', '代码', 'symbol', 'Symbol', '证券代码']
    candidates_name = ['股票简称', '名称', 'name', 'Name']
    code_col = next((c for c in candidates_code if c in df.columns), None)
    name_col = next((c for c in candidates_name if c in df.columns), None)
    if code_col is None or name_col is None:
        return {}
    codes = df[code_col].dropna().astype(str).str.strip()
    names = df[name_col].dropna().astype(str).str.strip()
    if len(codes) != len(df[name_col].dropna()):
        df = df[[code_col, name_col]].dropna()
        codes = df[code_col].astype(str).str.strip()
        names = df[name_col].astype(str).str.strip()
    codes = codes[codes.str.match(r'^\d{6}(\.(SZ|SH))?$', na=False)]
    mapping = {}
    for code, name in zip(codes, names):
        if '.' in code:
            key = code.upper()
        else:
            clean = code.zfill(6)
            key = f"{clean}.SH" if clean.startswith('60') else f"{clean}.SZ"
        mapping[key] = name
    return mapping

# ==================== 8. 主程序 ====================
class StrategyParams:
    def __init__(self):
        self.stocknum = 4
        self.buyagain = 5
        self.commission_rate = 0.0001
        self.min_commission = 0.01
        self.initial_capital = 1_0000.0
params = StrategyParams()


# 注意：要求时间间隔至少2个月，且第1周用于生成均线，不产生交易。
START_DATE = '2025-09-14'
END_DATE = '2025-11-14'
USE_NEW_DATA = False  # 是否下载新数据（True: 下载新数据并覆盖旧数据，False: 尝试使用缓存的数据）

# 股票池由Excel每日更新覆盖
STOCK_POOL = []
EXCEL_STOCK_POOL_PATH = r"f:\21.My_CodeBase\3.my_trade\5日平均成交额排名前100，主板，现价大于20日均....xlsx"
CODE_NAME_MAP = {}


if __name__ == '__main__':
    result = run_backtest()
    if result is not None:
        equity_curve, trade_log = result
        print("回测完成✓")
    else:
        print("回测失败✗")
