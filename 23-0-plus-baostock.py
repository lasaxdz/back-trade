# 单因子选股策略回测 (Baostock数据源)
# 回测期间: 2025-09-01 至 2025-11-01
# 初始资金: 10,000.00 元
# ============================================================
# 正在下载 18 只股票数据...
# 成功下载 18 只股票，失败 0 只
# 下载上证指数数据...
# 成功获取上证指数数据，共 39 条记录
# 开始回测...

# ================================================================================
# 交易交割单 (模拟)
# ================================================================================
# 交易日期         操作时间点      股票代码       操作           价格(元)      数量(股)
# --------------------------------------------------------------------------------
# 2025-09-30   收盘前        000063.SZ  买入           45.64      200.00
# 2025-10-14   收盘前        000063.SZ  破线           50.20      200.00
# 2025-10-16   收盘前        600089.SS  买入           20.07      100.00
# 2025-10-16   收盘前        601318.SS  买入           57.08      100.00
# 2025-10-16   收盘前        000568.SZ  买入           137.05     100.00
# 2025-10-17   收盘前        600089.SS  破线           18.89      100.00
# 2025-10-20   收盘前        000568.SZ  破线           133.15     100.00
# 2025-10-21   收盘前        601318.SS  清仓           57.14      100.00
# 2025-10-21   收盘前        000063.SZ  买入           51.24      100.00
# 2025-10-21   收盘前        601888.SS  买入           70.71      100.00
# 2025-10-22   收盘前        000063.SZ  破线           49.94      100.00
# 2025-10-22   收盘前        601888.SS  破线           69.67      100.00
# 2025-10-23   收盘前        603799.SS  买入           62.75      100.00
# 2025-10-24   收盘前        603799.SS  买入           62.98      100.00
# 2025-10-28   收盘前        603799.SS  破线           60.50      200.00
# 2025-10-28   收盘前        002475.SZ  买入           64.66      100.00
# 2025-10-28   收盘前        002384.SZ  买入           73.90      100.00
# 2025-10-28   收盘前        002460.SZ  买入           64.68      100.00
# 2025-10-28   收盘前        600089.SS  买入           19.19      100.00
# 2025-10-28   收盘前        601318.SS  买入           57.76      100.00
# 2025-10-28   收盘前        002466.SZ  买入           48.51      100.00
# 2025-10-29   收盘前        600089.SS  买入           20.90      100.00
# 2025-10-30   收盘前        002475.SZ  破线           65.04      100.00
# 2025-10-30   收盘前        002384.SZ  破线           72.65      100.00
# 2025-10-31   收盘前        600089.SS  破线           19.58      200.00
# 2025-10-31   收盘前        601318.SS  破线           57.83      100.00
# --------------------------------------------------------------------------------

# === 回测结果 ===
# 起始资金: 10,000.00 元
# 结束资金: 10,436.03 元
# 总收益率: 4.36%
# 上证指数基准收益率: 2.05%
# logout success!
# 登出Baostock系统

# ✓ 回测完成

# (money) F:\21.My_CodeBase\3.my_trade>




# -*- coding: utf-8 -*-
"""
单因子选股策略 → Baostock + pandas 回测 + 可视化 + 交割单
数据源：Baostock (http://baostock.com/)
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
    """
    将Yahoo Finance格式的股票代码转换为Baostock格式
    例如: 600438.SS -> sh.600438, 002130.SZ -> sz.002130
    """
    code = symbol.split('.')[0]
    if symbol.endswith('.SS'):
        return f'sh.{code}'
    elif symbol.endswith('.SZ'):
        return f'sz.{code}'
    return symbol



def get_stock_hist_data(symbol, start_date, end_date, adjustflag='2'):
    """
    使用Baostock获取单只股票历史数据
    adjustflag: '1'后复权, '2'前复权, '3'不复权
    """
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
def filter_gem_stock(stock_list):
    """只保留00和60开头的股票（主板）"""
    filtered = []
    for stock in stock_list:
        code = stock.split('.')[0]
        if code.startswith(('00', '60')):
            filtered.append(stock)
    return filtered

def filter_paused_stock(stock_list, all_data, current_date):
    """过滤停牌股票（用成交量判断）"""
    filtered = []
    for stock in stock_list:
        if stock not in all_data:
            continue
        
        df = all_data[stock]
        if current_date not in df.index:
            continue
        
        volume = df.loc[current_date, 'volume']
        if volume > 0:
            filtered.append(stock)
    
    return filtered

def filter_buyagain(stock_list, sold_stock):
    """过滤卖出不足params.buyagain日的股票"""
    return [stock for stock in stock_list if stock not in sold_stock]

def filter_price_above_ma5(stock_list, all_data, current_date):
    """过滤股价小于5日均线的个股"""
    filtered = []
    
    for stock in stock_list:
        if stock not in all_data:
            continue
        
        df = all_data[stock]
        if current_date not in df.index or len(df) < 5:
            continue
        
        current_close = df.loc[current_date, 'close']
        current_idx = df.index.get_loc(current_date)
        ma5_data = df.iloc[current_idx-4:current_idx+1]
        ma5 = ma5_data['close'].mean()
        
        if current_close >= ma5:
            filtered.append(stock)
    
    return filtered

def filter_price_above_ma20(stock_list, all_data, current_date):
    """过滤股价小于20日均线的个股"""
    filtered = []
    
    for stock in stock_list:
        if stock not in all_data:
            continue
        
        df = all_data[stock]
        if current_date not in df.index or len(df) < 20:
            continue
        
        current_close = df.loc[current_date, 'close']
        current_idx = df.index.get_loc(current_date)
        ma20_data = df.iloc[current_idx-19:current_idx+1]
        ma20 = ma20_data['close'].mean()
        
        if current_close >= ma20:
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

def is_price_below_ma5(symbol, all_data, current_date):
    """判断股价是否跌破5日均线"""
    if symbol not in all_data:
        return False
    
    df = all_data[symbol]
    if current_date not in df.index or len(df) < 5:
        return False
    
    current_idx = df.index.get_loc(current_date)
    if current_idx < 4:
        return False
    
    current_price = df.loc[current_date, 'close']
    ma5 = get_ma5(symbol, all_data, current_date)
    
    return current_price < ma5

# ==================== 4. 选股与排名函数 ====================
def get_stock_list(all_data, current_date, sold_stock, portfolio_positions, is_selection_day=False):
    """获取股票列表"""
    stock_list = STOCK_POOL.copy()
    
    stock_list = filter_gem_stock(stock_list)
    
    if is_selection_day:
        stock_list = filter_paused_stock(stock_list, all_data, current_date)
        stock_list = filter_price_above_ma5(stock_list, all_data, current_date)
        stock_list = filter_price_above_ma20(stock_list, all_data, current_date)
        stock_list = filter_turnover_below_avg(stock_list, all_data, current_date)
    
    stock_list = filter_buyagain(stock_list, sold_stock)
    
    return stock_list

def get_volume_5d_turnover(symbol, all_data, current_date):
    """获取5日平均成交金额"""
    if symbol not in all_data:
        return 0
    
    df = all_data[symbol]
    try:
        current_idx = df.index.get_loc(current_date)
    except:
        return 0
    
    if current_idx < 4:
        return 0
    
    start_idx = current_idx - 4
    end_idx = current_idx + 1
    recent_data = df.iloc[start_idx:end_idx]
    
    # Baostock的amount字段就是成交金额
    if 'turnover_money' in recent_data.columns:
        return recent_data['turnover_money'].mean()
    else:
        turnover_money = recent_data['close'] * recent_data['volume']
        return turnover_money.mean()

def filter_turnover_below_avg(stock_list, all_data, current_date):
    """过滤当日成交金额低于5日平均成交金额的股票"""
    filtered = []
    
    for stock in stock_list:
        if stock not in all_data:
            continue
        
        df = all_data[stock]
        if current_date not in df.index or len(df) < 5:
            continue
        
        # 当日成交金额
        if 'turnover_money' in df.columns:
            current_turnover = df.loc[current_date, 'turnover_money']
        else:
            current_price = df.loc[current_date, 'close']
            current_volume = df.loc[current_date, 'volume']
            current_turnover = current_price * current_volume
        
        # 5日平均成交金额
        avg_turnover_5d = get_volume_5d_turnover(stock, all_data, current_date)
        
        if avg_turnover_5d > 0 and current_turnover < avg_turnover_5d:
            filtered.append(stock)
    
    return filtered

def get_stock_rank_m_m(all_data, stock_list, current_date):
    """股票排名 - 单因子：5日平均成交金额"""
    if not stock_list:
        return []
    
    ranked_stocks = []
    turnover_5d_list = []
    
    for symbol in stock_list:
        if symbol not in all_data:
            continue
        
        df = all_data[symbol]
        if current_date not in df.index:
            continue
        
        turnover_5d = get_volume_5d_turnover(symbol, all_data, current_date)
        
        if turnover_5d > 0:
            turnover_5d_list.append(turnover_5d)
            ranked_stocks.append(symbol)
    
    if not ranked_stocks:
        return []
    
    scores = [(ranked_stocks[i], turnover_5d_list[i]) for i in range(len(ranked_stocks))]
    scores.sort(key=lambda x: x[1], reverse=True)
    
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
    
    # 添加操作时间点列，默认为"收盘前"，因为策略是基于日线数据
    headers = ["交易日期", "操作时间点", "股票代码", "操作", "价格(元)", "数量(股)"]
    print("{:<12} {:<10} {:<10} {:<12} {:<10} {:<12}".format(*headers))
    print("-" * 80)
    
    for trade in trade_log:
        date_str = trade['date'].strftime('%Y-%m-%d')
        # 默认交易时间点为收盘前，因为这是基于日线数据的回测
        time_point = "收盘前"
        symbol = trade['symbol']
        action = trade['action']
        price = trade['price']
        shares = trade['shares']
        
        print(f"{date_str:<12} {time_point:<10} {symbol:<10} {action:<12} {price:<10.2f} {shares:<12.2f}")
    
    print("-" * 80)

# ==================== 6. 回测主循环 ====================
def run_backtest():
    """运行回测"""
    # 登录Baostock系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f'登录失败: {lg.error_msg}')
        return None, None
    print('登录Baostock成功')
    
    try:
        print("=" * 60)
        print(f"单因子选股策略回测 (Baostock数据源)")
        print(f"回测期间: {START_DATE} 至 {END_DATE}")
        print(f"初始资金: {params.initial_capital:,.2f} 元")
        print("=" * 60)
        
        # 1. 下载股票数据
        all_data, failed_symbols = get_all_stock_data(STOCK_POOL, START_DATE, END_DATE)
        
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
        
        candidate_stocks = []
        
        # 4. 回测主循环
        print("开始回测...")
        total_days = len(dates)
        for i, current_date in enumerate(dates):
            if (i + 1) % (total_days // 10) == 0 or i == total_days - 1:
                progress = (i + 1) / total_days * 100
                # print(f"回测进度: {progress:.1f}% ({i+1}/{total_days})")
            
            # 更新卖出计数器
            to_remove = []
            for stock in sold_stock:
                sold_stock[stock] -= 1
                if sold_stock[stock] <= 0:
                    to_remove.append(stock)
            for stock in to_remove:
                del sold_stock[stock]
            
            # 检查是否为选股日
            is_selection_day = current_date.weekday() in [1, 3]
            
            if is_selection_day:
                stock_candidates = get_stock_list(all_data, current_date, sold_stock, portfolio.keys(), is_selection_day=True)
                ranked_stocks = get_stock_rank_m_m(all_data, stock_candidates, current_date)
                candidate_stocks = ranked_stocks
            
            # 1. 首先检查所有持仓股票是否跌破5日均线，如果跌破则卖出（每个交易日都执行）
            for stock in list(portfolio.keys()):
                # 只检查是否跌破5日均线，这是一个独立的卖出条件
                if is_price_below_ma5(stock, all_data, current_date):
                    if stock not in all_data or current_date not in all_data[stock].index:
                        continue
                    
                    price = all_data[stock].loc[current_date, 'close']
                    shares = portfolio[stock]
                    if shares > 0:
                        sell_reason = '破线'
                        
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
            
            # 2. 选股日调仓换股（卖出不在候选池中且未被卖出的股票，买入新的股票）
            if candidate_stocks:
                total_value = cash + sum(portfolio[s] * (all_data[s].loc[current_date, 'close'] if s in all_data and current_date in all_data[s].index else 0) for s in portfolio)
                
                target_stocknum = min(params.stocknum, len(candidate_stocks))
                if target_stocknum > 0:
                    buycash_per_stock = total_value / target_stocknum
                else:
                    buycash_per_stock = 0
                
                # 卖出不在候选股票池中的股票（已经检查过跌破3日均线的情况）
                for stock in list(portfolio.keys()):
                    if stock not in candidate_stocks:
                        if stock not in all_data or current_date not in all_data[stock].index:
                            continue
                        
                        price = all_data[stock].loc[current_date, 'close']
                        shares = portfolio[stock]
                        if shares > 0:
                            sell_reason = '清仓'
                            
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
                
                # 买入
                buy_list = [s for s in candidate_stocks[:10] if s not in sold_stock]
                
                for stock in buy_list:
                    if stock not in all_data or current_date not in all_data[stock].index:
                        continue
                    
                    price = all_data[stock].loc[current_date, 'close']
                    if price <= 0:
                        continue
                    
                    current_shares = portfolio.get(stock, 0)
                    current_value = current_shares * price
                    
                    target_value = buycash_per_stock
                    
                    if current_value >= target_value * 0.9:
                        continue
                    
                    tobuy_value = target_value - current_value
                    if tobuy_value <= 0:
                        continue
                    
                    shares_to_buy = int(tobuy_value / (price * (1 + params.commission_rate)))
                    shares_to_buy = max(100, (shares_to_buy // 100) * 100)
                    
                    if shares_to_buy <= 0:
                        continue
                    
                    buy_amount = shares_to_buy * price
                    commission = max(buy_amount * params.commission_rate, params.min_commission)
                    total_cost = buy_amount + commission
                    
                    if total_cost > cash * 0.99:
                        shares_to_buy = int(cash * 0.99 / (price * (1 + params.commission_rate)))
                        shares_to_buy = max(100, (shares_to_buy // 100) * 100)
                        
                        if shares_to_buy <= 0:
                            continue
                        
                        buy_amount = shares_to_buy * price
                        commission = max(buy_amount * params.commission_rate, params.min_commission)
                        total_cost = buy_amount + commission
                    
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

# ==================== 8. 主程序 ====================
class StrategyParams:
    def __init__(self):
        self.stocknum = 4
        self.buyagain = 5
        self.commission_rate = 0.0001
        self.min_commission = 0.01
        self.initial_capital = 10000.0

params = StrategyParams()

START_DATE = '2025-09-01'
END_DATE = '2025-11-01'

STOCK_POOL = [
    "601138.SS", "601012.SS", "600089.SS", "601888.SS", "603799.SS", 
    "601899.SS", "601318.SS", "600111.SS", "600438.SS", 
    "002475.SZ", "002466.SZ", "002460.SZ", "000002.SZ", "000858.SZ", 
    "000063.SZ", "002384.SZ", "000568.SZ", "000792.SZ"
]

if __name__ == '__main__':
    result = run_backtest()
    if result is not None:
        equity_curve, trade_log = result
        print("\n✓ 回测完成")
    else:
        print("\n✗ 回测失败")
