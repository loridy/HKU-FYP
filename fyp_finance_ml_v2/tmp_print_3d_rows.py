import pandas as pd
m=pd.read_csv('outputs/metrics/02_live_metrics.csv')
sub=m[m.horizon_days==3]
print('n rows',len(sub))
print(sub[['feature_set','model','n_train_rows','n_val_rows','n_test_rows','train_start','train_end','val_start','val_end','test_start','test_end']].drop_duplicates().to_string(index=False))
