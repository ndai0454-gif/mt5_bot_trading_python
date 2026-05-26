import pandas as pd
import numpy as np
from datetime import datetime

def analyze():
    try:
        df = pd.read_csv('backtest_results.csv')
    except Exception as e:
        print(f'Error reading CSV: {e}')
        return

    print('='*40)
    print('   DETAILED BACKTEST ANALYSIS')
    print('='*40)

    # 1. General Stats
    total_trades = len(df)
    # Phân tích dựa trên cột PnL
    if 'PnL' in df.columns:
        wins = len(df[df['PnL'] > 0])
        losses = len(df[df['PnL'] < 0])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = df['PnL'].sum()
    else:
        # fallback nếu chỉ có cột Result
        wins = len(df[df['Result'] == 'WIN']) if 'Result' in df.columns else 0
        losses = len(df[df['Result'] == 'LOSS']) if 'Result' in df.columns else 0
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = 0

    print(f'Total Trades: {total_trades}')
    print(f'Overall Win Rate: {win_rate:.2f}%')
    if 'PnL' in df.columns:
        print(f'Total Net PnL: ${total_pnl:.2f}')

    # 2. H1 Trend Efficiency
    if 'H1_Trend' in df.columns and 'PnL' in df.columns:
        print('\n--- PERFORMANCE BY H1 TREND ---')
        trend_groups = df.groupby('H1_Trend')
        for trend, group in trend_groups:
            t_wins = len(group[group['PnL'] > 0])
            t_total = len(group)
            t_wr = (t_wins / t_total) * 100
            t_pnl = group['PnL'].sum()
            print(f'{trend:10} | WinRate: {t_wr:6.2f}% | PnL: ${t_pnl:10.2f} | Trades: {t_total}')

    # 3. Time Analysis
    if 'Time' in df.columns and 'PnL' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df['hour'] = df['Time'].dt.hour
        print('\n--- PROFITABILITY BY HOUR (UTC) ---')
        hour_stats = df.groupby('hour')['PnL'].sum().sort_values(ascending=False)
        print(hour_stats.head(5))

    # 4. R:R Analysis
    if 'PnL' in df.columns:
        avg_win = df[df['PnL'] > 0]['PnL'].mean()
        avg_loss = abs(df[df['PnL'] < 0]['PnL'].mean())
        if avg_loss > 0:
            rr_ratio = avg_win / avg_loss
            print(f'\n--- RISK REWARD RATIO ---')
            print(f'Avg Win: ${avg_win:.2f}')
            print(f'Avg Loss: ${avg_loss:.2f}')
            print(f'Realized R:R: {rr_ratio:.2f}')

if __name__ == '__main__':
    analyze()