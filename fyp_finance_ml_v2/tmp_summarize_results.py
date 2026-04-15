import pandas as pd
from pathlib import Path

m = pd.read_csv(Path('outputs/metrics/02_live_metrics.csv'))
b = pd.read_csv(Path('outputs/metrics/02_live_backtest_summary.csv'))
r = pd.read_csv(Path('outputs/metrics/02_live_relative_summary.csv'))

df = m.merge(b, on=['notebook_tag','mode','feature_set','feature_groups','model','n_features_used','horizon_days'], how='left', suffixes=('','_bt'))
df = df.merge(r[['feature_set','model','horizon_days','alpha_ann','excess_ann_return','information_ratio']], on=['feature_set','model','horizon_days'], how='left')

# helper: show top configs within each horizon
for h in sorted(df['horizon_days'].unique()):
    sub = df[df['horizon_days']==h].copy()
    print('\n==== HORIZON', h, 'D ====')
    # top by balanced accuracy
    top_ba = sub.sort_values('balanced_accuracy', ascending=False).head(5)
    print('\nTop 5 by balanced_accuracy:')
    print(top_ba[['feature_set','model','balanced_accuracy','accuracy','roc_auc','rank_ic','icir','top_k_hit_rate']].to_string(index=False))

    # top by rank_ic
    top_ic = sub.sort_values('rank_ic', ascending=False).head(5)
    print('\nTop 5 by rank_ic:')
    print(top_ic[['feature_set','model','rank_ic','icir','balanced_accuracy','roc_auc','top_bottom_spread','bucket_monotonicity']].to_string(index=False))

    # top by sharpe
    top_sh = sub.sort_values('sharpe', ascending=False).head(5)
    print('\nTop 5 by sharpe (execution-aware backtest):')
    print(top_sh[['feature_set','model','sharpe','cumulative_return','annualized_return','max_drawdown','avg_turnover','avg_cost_drag','backtest_method']].to_string(index=False))

# single best overall sharpe
best = df.sort_values('sharpe', ascending=False).head(10)
print('\n==== TOP 10 SHARPE OVERALL ====')
print(best[['horizon_days','feature_set','model','sharpe','annualized_return','max_drawdown','avg_turnover','avg_cost_drag']].to_string(index=False))
