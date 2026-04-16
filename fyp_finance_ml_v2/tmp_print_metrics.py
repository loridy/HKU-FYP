import pandas as pd
from pathlib import Path
pd.set_option('display.max_columns', 200)
base = Path('outputs/metrics')
for fn in ['02_live_metrics.csv','02_live_backtest_summary.csv','02_live_relative_summary.csv']:
    df = pd.read_csv(base/fn)
    print('\n===', fn, '===')
    print('shape', df.shape)
    print('columns', list(df.columns))
    print(df.head(10).to_string(index=False))
