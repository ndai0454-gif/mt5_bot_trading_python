import pandas as pd

df = pd.read_csv('backtest_results.csv')

# 1. General Stats
total_trades = df[df['Result'] != 'OPEN'].shape[0]
wins = df[df['Result'] == 'WIN'].shape[0]
losses = df[df['Result'] == 'LOSS'].shape[0]
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
total_pnl = df['PnL'].sum()

# 2. Performance by Direction
direction_stats = df[df['Result'] != 'OPEN'].groupby('Direction')['PnL'].agg(['sum', 'count', 'mean'])
direction_stats['win_rate'] = df[df['Result'] == 'WIN'].groupby('Direction').size() / df[df['Result'] != 'OPEN'].groupby('Direction').size() * 100
direction_stats = direction_stats.fillna(0)

# 3. Performance by Regime
regime_stats = df[df['Result'] != 'OPEN'].groupby('Regime')['PnL'].agg(['sum', 'count', 'mean'])
regime_stats['win_rate'] = df[df['Result'] == 'WIN'].groupby('Regime').size() / df[df['Result'] != 'OPEN'].groupby('Regime').size() * 100
regime_stats = regime_stats.fillna(0)

# 4. Performance by H1 Trend
trend_stats = df[df['Result'] != 'OPEN'].groupby('H1_Trend')['PnL'].agg(['sum', 'count', 'mean'])
trend_stats['win_rate'] = df[df['Result'] == 'WIN'].groupby('H1_Trend').size() / df[df['Result'] != 'OPEN'].groupby('H1_Trend').size() * 100
trend_stats = trend_stats.fillna(0)

print("--- General Stats ---")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total PnL: ${total_pnl:.2f}")
print("\n--- Direction Stats ---")
print(direction_stats)
print("\n--- Regime Stats ---")
print(regime_stats)
print("\n--- Trend Stats ---")
print(trend_stats)
