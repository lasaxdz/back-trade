# -*- coding: utf-8 -*-
"""
最小化策略回测示例 (Baostock 数据源)
策略要点：
- 多周期 KDJ 共振：月/周/日三周期均为 K>D
- 日线 MACD 金叉: DIF>DEA (动量向上)
- 均线强趋势: MA5>MA10 且两者斜率角度分别>=20、>=10
- 量能 MACD 近期金叉: 近 0-10 个交易日出现 DIF1 上穿 DEA1
- 卖出: 收盘价跌破 MA5
"""
import baostock as bs
import pandas as pd
import numpy as np
import argparse

def ema(s, n):
    """指数移动平均 (EMA): 使用 `span=n` 且 `adjust=False`,与常见技术分析口径保持一致。"""
    return s.ewm(span=int(n), adjust=False).mean()

def kdj(df, n=9, m1=3, m2=3):
    """计算 K、D 值 (不含 J)
    - RSV = (C - LLV(n)) / (HHV(n) - LLV(n)) * 100
    - K = EMA(RSV, m1)
    - D = EMA(K, m2)
    返回包含 K、D 两列的 DataFrame。
    """
    ll = df['low'].rolling(int(n)).min()
    hh = df['high'].rolling(int(n)).max()
    rsv = (df['close'] - ll) / (hh - ll) * 100
    k = ema(rsv.fillna(0), m1)
    d = ema(k, m2)
    return pd.DataFrame({'K': k, 'D': d}, index=df.index)

def resample_ohlc(df, rule):
    """按 OHLCV 规则重采样到周/月线 """
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return df.resample(rule).apply(agg).dropna()

def get_data(code, start_date, end_date):
    """拉取区间日线 (前复权), 过滤停牌, 返回 DataFrame """
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(lg.error_msg)
    rs = bs.query_history_k_data_plus(
        code,
        'date,open,high,low,close,volume,amount,tradestatus',
        start_date=start_date,
        end_date=end_date,
        frequency='d',
        adjustflag='2'  # 2=前复权，常用于回测
    )
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open','high','low','close','volume','amount','tradestatus']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # 仅保留交易状态为 1（正常交易）的记录
    df = df[df['tradestatus'] == 1]
    df = df[['date','open','high','low','close','volume']].set_index('date').sort_index()
    return df


def run(symbol='sh.601138', index_code='sh.000001', start=None, end=None,
        SHORT=4, LONG=10, MID=3, SHORT1=4, LONG1=10, MID1=3):
    """最小化回测流程: 单标的, 满仓买入/跌破 MA5 卖出。若样本不足 100 交易日, 会自动扩大时间范围保证样本量。"""
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    if start is None:
        start = (pd.Timestamp(end) - pd.Timedelta(days=720)).strftime('%Y-%m-%d')
    df = get_data(symbol, start, end)
    if len(df) < 100:
        start = (pd.Timestamp(end) - pd.Timedelta(days=1500)).strftime('%Y-%m-%d')
        df = get_data(symbol, start, end)
    if len(df) < 100:
        raise RuntimeError('样本交易日不足 100 天，请更换标的或扩大时间范围')
    idx = get_data(index_code, start, end)
    if df.empty or idx.empty:
        raise RuntimeError('数据获取失败')

    # 计算日/周/月 KDJ，并将周/月指标回填到日频对齐
    day_kd = kdj(df)
    wk = resample_ohlc(df, 'W-FRI')
    mo = resample_ohlc(df, 'M')
    wk_kd = kdj(wk)
    mo_kd = kdj(mo)
    wk_kd = wk_kd.reindex(df.index, method='ffill')
    mo_kd = mo_kd.reindex(df.index, method='ffill')

    # 日线 MACD：DIF 与 DEA（EMA 形式）
    dif = ema(df['close'], SHORT) - ema(df['close'], LONG)
    dea = ema(dif, MID)
    t2 = dif > dea

    # 均线与斜率（角度）：MA5、MA10 以及对应的斜率角度
    ma5 = df['close'].rolling(5).mean()
    ma10 = df['close'].rolling(10).mean()
    slope5 = np.arctan((ma5 / ma5.shift(1) - 1) * 100) * 57.3
    slope10 = np.arctan((ma10 / ma10.shift(1) - 1) * 100) * 57.3
    t3 = (ma5 > ma10) & (ma5 > ma5.shift(1)) & (slope5 >= 20) & (slope10 >= 10)

    # 量能 MACD：近 0-10 个交易日内是否出现 DIF1 上穿 DEA1（金叉）
    dif1 = ema(df['volume'], SHORT1) - ema(df['volume'], LONG1)
    dea1 = ema(dif1, MID1)
    cross_up = (dif1 > dea1) & (dif1.shift(1) <= dea1.shift(1))
    pos = np.arange(len(cross_up))
    last_pos = pd.Series(np.where(cross_up, pos, np.nan), index=cross_up.index).ffill()
    days_since = pd.Series(pos, index=cross_up.index) - last_pos
    days_since = days_since.fillna(1e9)
    t4 = (days_since >= 0) & (days_since <= 10)

    # 聚合买卖信号
    t1 = (mo_kd['K'] > mo_kd['D']) & (wk_kd['K'] > wk_kd['D']) & (day_kd['K'] > day_kd['D'])
    buy = t1 & t2 & t3 & t4
    sell = df['close'] < ma5

    # 资金曲线（最小化）：满仓/空仓切换
    cash = 1_00000.0
    shares = 0.0
    equity = []
    holding = False
    for dt in df.index:
        price = df.loc[dt, 'close']
        if not holding and buy.loc[dt]:
            shares = cash / price  # 满仓买入
            cash = 0.0
            holding = True
        elif holding and sell.loc[dt]:
            cash = shares * price  # 跌破 MA5 卖出
            shares = 0.0
            holding = False
        equity.append(cash + shares * price)
    equity = pd.Series(equity, index=df.index)

    # 输出收益率对比：策略 vs 上证指数
    strat_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    idx_return = (idx['close'].iloc[-1] / idx['close'].iloc[0] - 1) * 100
    print(f'样本交易日数: {len(df)}')
    print(f'策略总收益率: {strat_return:.2f}%')
    print(f'上证指数收益率: {idx_return:.2f}%')

def main():
    """命令行入口：支持标的、指数与各周期参数调节。"""
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='sh.601138')  # 回测标的示例
    p.add_argument('--index', default='sh.000001')   # 上证指数用于对比
    p.add_argument('--start')                        # 起始日期 YYYY-MM-DD
    p.add_argument('--end')                          # 结束日期 YYYY-MM-DD
    p.add_argument('--SHORT', type=int, default=4)
    p.add_argument('--LONG', type=int, default=10)
    p.add_argument('--MID', type=int, default=3)
    p.add_argument('--SHORT1', type=int, default=4)
    p.add_argument('--LONG1', type=int, default=10)
    p.add_argument('--MID1', type=int, default=3)
    args = p.parse_args()
    run(args.symbol, args.index, args.start, args.end,
        args.SHORT, args.LONG, args.MID, args.SHORT1, args.LONG1, args.MID1)

if __name__ == '__main__':
    main()
