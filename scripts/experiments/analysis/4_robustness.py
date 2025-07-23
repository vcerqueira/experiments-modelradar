import warnings

import pandas as pd
from utilsforecast.losses import smape
from modelradar.evaluate.radar import ModelRadar

from utils.load_data.config import DATA_GROUPS

warnings.filterwarnings('ignore')
EXPERIMENT = 'robust-nf'

results_list = []
for data_name, group in DATA_GROUPS:
    print(data_name, group)
    df = pd.read_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv')
    df['Dataset'] = f'{data_name},{group}'

    results_list.append(df)

results_df = pd.concat(results_list)

metadata = ['unique_id', 'ds', 'y', '-hi', '-lo',
            'trend_str', 'seas_str', 'Dataset',
            'is_anomaly', 'large_obs',
            'large_uids', 'anomaly_status', 'Frequency', 'seed']
model_names = results_df.columns[~results_df.columns.str.contains('|'.join(metadata))].tolist()

radar = ModelRadar(cv_df=results_df,
                   metrics=[smape],
                   model_names=model_names,
                   hardness_reference='AutoMLP',
                   ratios_reference='AutoMLP',
                   rope=10)

err_df_ = radar.evaluate_by_group(group_col='seed')

err_melted = err_df_.T.reset_index().melt('index')

# std_data = {}
# for g, df_ in results_df.groupby('Dataset'):
#     print(g)
#     err_df_ = radar.evaluate_by_group(group_col='seed', cv=df_)
#     std_data[g] = err_df_.std(axis=1)
#
# pd.DataFrame(std_data)
