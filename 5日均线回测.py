# 准备工作:
# pip install pandas numpy matplotlib yfinance

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 设置中文字体，解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 配置代理
proxy = "http://127.0.0.1:7897"
yf.set_config(proxy=proxy)

# 日期范围设置
end_date = "2025-09-24"
start_date = "2025-06-01"

# 股票选择
stock_symbol = "002237.Sz"
print(f"正在分析股票: {stock_symbol}")

# 获取股票数据
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# 数据有效性检查
if stock_data.empty:
    print("错误: 未获取到股票数据，请检查股票代码和日期范围")
    exit()

print(f"数据获取成功 - 共 {len(stock_data)} 条记录")
print(f"数据时间范围: {stock_data.index[0].date()} 至 {stock_data.index[-1].date()}")

# 处理可能的多级列索引
if isinstance(stock_data.columns, pd.MultiIndex):
    print("处理多级列索引...")
    stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

# 确定所需价格列
close_price_column = next((col for col in stock_data.columns if 'Close' in col or 'close' in col), None)
open_price_column = next((col for col in stock_data.columns if 'Open' in col or 'open' in col), None)
high_price_column = next((col for col in stock_data.columns if 'High' in col or 'high' in col), None)
low_price_column = next((col for col in stock_data.columns if 'Low' in col or 'low' in col), None)

# 确保找到所有必要的价格列
if None in [close_price_column, open_price_column, high_price_column, low_price_column]:
    print("警告: 无法找到所有所需的价格列，使用默认列名")
    close_price_column = 'Close' if 'Close' in stock_data.columns else stock_data.columns[3]
    open_price_column = 'Open' if 'Open' in stock_data.columns else stock_data.columns[0]
    high_price_column = 'High' if 'High' in stock_data.columns else stock_data.columns[1]
    low_price_column = 'Low' if 'Low' in stock_data.columns else stock_data.columns[2]

# 计算移动平均线 - 5日均线（用于买入和卖出条件）【核心修改：window从3改为5】
stock_data['5日均线'] = stock_data[close_price_column].rolling(window=5, min_periods=1).mean()

# --------------- 定义买入和卖出条件 ---------------
# 初始化信号列：0表示无信号，1表示买入，-1表示卖出
stock_data['交易信号'] = 0

# 确定有效数据范围（均线有值）【核心修改：检查5日均线有效性】
valid_data_mask = stock_data['5日均线'].notna()

# 买入条件：收盘价大于5日均线，且前一天收盘价不大于5日均线（避免连续买入）【核心修改：替换为5日均线】
buy_condition = (stock_data[close_price_column] > stock_data['5日均线']) & \
                (stock_data[close_price_column].shift(1) <= stock_data['5日均线'].shift(1)) & \
                valid_data_mask

# 卖出条件：收盘价低于5日均线，且前一天收盘价不低于5日均线（避免连续卖出）【核心修改：替换为5日均线】
sell_condition = (stock_data[close_price_column] < stock_data['5日均线']) & \
                 (stock_data[close_price_column].shift(1) >= stock_data['5日均线'].shift(1)) & \
                 valid_data_mask

# 应用交易信号
stock_data.loc[buy_condition, '交易信号'] = 1    # 买入信号
stock_data.loc[sell_condition, '交易信号'] = -1   # 卖出信号

# 根据最新信号确定持仓状态：1表示持有（多头），0表示不持有
stock_data['持仓状态'] = stock_data['交易信号'].replace(0, method='ffill')
stock_data['持仓状态'].fillna(0, inplace=True)  # 初始状态为0（无持仓）
stock_data['持仓状态'] = stock_data['持仓状态'].clip(0, 1)  # 只允许0或1，不考虑空头

# ------------------------------------------------------

# 识别交易对（买入后跟随卖出）
buy_signals = stock_data[stock_data['交易信号'] == 1].index
sell_signals = stock_data[stock_data['交易信号'] == -1].index

# 匹配买入和卖出信号，形成完整交易
transactions = []
current_buy = None
last_date = stock_data.index[-1]  # 数据最后一天
last_price = stock_data.loc[last_date, close_price_column]  # 最后一天收盘价

for date in stock_data.index:
    if date in buy_signals and current_buy is None:
        # 记录买入价格和日期
        current_buy = {
            'date': date,
            'price': stock_data.loc[date, close_price_column]
        }
    elif date in sell_signals and current_buy is not None:
        # 计算这笔交易的收益率
        sell_price = stock_data.loc[date, close_price_column]
        return_rate = (sell_price - current_buy['price']) / current_buy['price']
        
        # 记录完整交易
        transactions.append({
            '买入日期': current_buy['date'],
            '买入价格': current_buy['price'],
            '卖出日期': date,
            '卖出价格': sell_price,
            '收益率': return_rate,
            '状态': '已平仓'
        })
        
        current_buy = None

# 处理最后仍持有的情况，计算至今的收益率
if current_buy is not None:
    # 计算从买入到最后一天的收益率
    return_rate = (last_price - current_buy['price']) / current_buy['price']
    
    transactions.append({
        '买入日期': current_buy['date'],
        '买入价格': current_buy['price'],
        '卖出日期': last_date,  # 使用最后一天作为卖出日期
        '卖出价格': last_price,
        '收益率': return_rate,
        '状态': '未平仓(至今)'
    })
    
    print(f"\n注意：最后一笔买入({current_buy['date'].date()})尚未卖出，已计算至最后交易日({last_date.date()})的收益率")

# 计算所有交易的收益率总和（复利计算，符合投资收益逻辑）
total_return = 1.0
if transactions:
    for t in transactions:
        total_return *= (1 + t['收益率'])
    total_return -= 1  # 转换为相对于初始本金的总收益率

# 创建单一图形（K线图与交易信号）
fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))
fig.suptitle("K线图与5日均线策略交易信号（含收益率总和）", fontsize=18)  # 标题修改为5日均线

# 绘制K线图
# 上涨用红色，下跌用绿色
up = stock_data[stock_data[close_price_column] >= stock_data[open_price_column]]
down = stock_data[stock_data[close_price_column] < stock_data[open_price_column]]

# 绘制K线实体
width = 0.6
width2 = 0.05
ax1.bar(up.index, up[close_price_column]-up[open_price_column], width, bottom=up[open_price_column], color='red')
ax1.bar(up.index, up[high_price_column]-up[low_price_column], width2, bottom=up[low_price_column], color='red')
ax1.bar(down.index, down[close_price_column]-down[open_price_column], width, bottom=down[open_price_column], color='green')
ax1.bar(down.index, down[high_price_column]-down[low_price_column], width2, bottom=down[low_price_column], color='green')

# 绘制5日均线【核心修改：替换为5日均线】
ax1.plot(stock_data.index, stock_data['5日均线'], label='5日均线', color='blue', linewidth=1.5)

# 标记买入信号
buy_dates = [t['买入日期'] for t in transactions]
ax1.scatter(buy_dates, 
           [stock_data.loc[t['买入日期'], high_price_column] * 1.02 for t in transactions], 
           marker='^', color='purple', label='买入信号', zorder=3)

# 标记卖出信号，区分已平仓和未平仓
for t in transactions:
    if t['状态'] == '已平仓':
        ax1.scatter(t['卖出日期'], 
                   stock_data.loc[t['卖出日期'], low_price_column] * 0.98, 
                   marker='v', color='black', label='卖出信号' if t == transactions[0] else "", zorder=3)
    else:
        ax1.scatter(t['卖出日期'], 
                   stock_data.loc[t['卖出日期'], low_price_column] * 0.98, 
                   marker='d', color='orange', label='未平仓(至今)' if t == transactions[-1] else "", zorder=3)

# 在买卖点之间绘制连线并标注单笔收益率
for i, transaction in enumerate(transactions):
    buy_date = transaction['买入日期']
    sell_date = transaction['卖出日期']
    return_rate = transaction['收益率']
    status = transaction['状态']
    
    # 绘制买卖点之间的连线，未平仓交易使用虚线
    line_style = '--' if status == '未平仓(至今)' else '-'
    ax1.plot([buy_date, sell_date], 
             [stock_data.loc[buy_date, close_price_column], 
              stock_data.loc[sell_date, close_price_column]], 
             'gray', linestyle=line_style, alpha=0.6)
    
    # 标注该笔交易的收益率
    mid_date = buy_date + (sell_date - buy_date) / 2
    mid_price = (stock_data.loc[buy_date, close_price_column] + 
                stock_data.loc[sell_date, close_price_column]) / 2
    
    # 未平仓交易使用不同背景色
    bbox_color = 'lightblue' if status == '未平仓(至今)' else 'yellow'
    ax1.annotate(f"{return_rate:.2%}", 
                (mdates.date2num(mid_date), mid_price),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', fc=bbox_color, alpha=0.7))

# 在图中右上角标注收益率总和（复利计算）
if transactions:
    x_pos = stock_data.index[-1] - (stock_data.index[-1] - stock_data.index[0]) * 0.02
    y_pos = stock_data[high_price_column].max() * 0.95
    
    ax1.annotate(
        f'所有交易总收益率\n(含未平仓): {total_return:.2%}',
        xy=(x_pos, y_pos),
        xytext=(0, 0),
        textcoords='offset points',
        ha='right',
        va='top',
        fontsize=12,
        weight='bold',
        bbox=dict(boxstyle='round,pad=1', fc='red', alpha=0.8, edgecolor='black'),
        color='white'
    )

# 设置K线图属性
ax1.set_title("K线图与交易信号（5日均线策略）", fontsize=14)  # 标题修改为5日均线
ax1.set_xlabel("日期", fontsize=12)
ax1.set_ylabel("价格", fontsize=12)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 显示每笔交易的收益率
if transactions:
    print("\n===== 每笔交易收益率 =====")
    transactions_df = pd.DataFrame(transactions)
    # 格式化日期和数字显示
    transactions_df['买入日期'] = transactions_df['买入日期'].dt.date
    transactions_df['卖出日期'] = transactions_df['卖出日期'].dt.date
    transactions_df['买入价格'] = transactions_df['买入价格'].round(2)
    transactions_df['卖出价格'] = transactions_df['卖出价格'].round(2)
    transactions_df['收益率'] = transactions_df['收益率'].apply(lambda x: f"{x:.2%}")
    print(transactions_df)

    # 计算交易成功次数（含未平仓的浮动盈利）
    success_count = sum(1 for t in transactions if t['收益率'] > 0)
    
    print(f"\n===== 交易结果汇总 =====")
    print(f"总交易次数: {len(transactions)}")
    closed_count = sum(1 for t in transactions if t['状态'] == '已平仓')
    open_count = len(transactions) - closed_count
    print(f"已平仓交易: {closed_count} 笔")
    print(f"未平仓交易: {open_count} 笔")
    print(f"交易成功次数: {success_count} 笔")
    print(f"所有交易总收益率(含未平仓，复利): {total_return:.2%}")
else:
    print("\n没有交易记录")