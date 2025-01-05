import pandas as pd
from modelradar.pipelines.utils import DecompositionSTL

from utils.load_data.config import DATASETS, DATA_GROUPS, GROUP_IDX

# ---- data loading and partitioning
EXPERIMENT = 'features'
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)

train, _ = data_loader.train_test_split(df, horizon=horizon)

# ---- feature extraction
strength_df = train.groupby('unique_id').apply(lambda x: DecompositionSTL.get_strengths(x, period=freq_int))
strength_df = pd.DataFrame.from_records(strength_df, index=strength_df.index)
strength_df['trend_str'] = (strength_df['trend_str'] > 0.6).map({False: 'No trend', True: 'With trend'})
strength_df['seasonal_str'] = (strength_df['seasonal_str'] > 0.6).map(
    {False: 'No seasonality', True: 'With seasonality'})
strength_df = strength_df.reset_index()

strength_df.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=False)
