import warnings

import pandas as pd
from neuralforecast import NeuralForecast
from utilsforecast.losses import smape
from modelradar.evaluate.radar import ModelRadar

from utils.load_data.config import DATASETS, DATA_GROUPS
from utils.models_config import ModelsConfig

warnings.filterwarnings('ignore')

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# ---- data loading and partitioning
SEEDS = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]
GROUP_IDX = 0
EXPERIMENT = 'robust-nf'
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

best_configs = pd.read_csv(f'assets/results/{data_name},{group},hpo-nf.csv').set_index('parameter')
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=5)
# df = data_loader.get_uid_tails(df, tail_size=100)
# df = data_loader.dummify_series(df)


train, test = data_loader.train_test_split(df, horizon=horizon)

results_list = []
for seed_ in SEEDS:
    # ---- model setup
    nf = NeuralForecast(models=ModelsConfig.get_nf_models_config(horizon=horizon,
                                                                 best_configs=best_configs,
                                                                 random_seed=seed_),
                        freq=freq_str)

    # ---- model fitting
    nf.fit(df=train)

    # ---- forecasts
    fcst_nf = nf.predict()
    fcst_nf = fcst_nf.merge(test, on=['ds', 'unique_id'], how='right')
    fcst_nf['seed'] = seed_

    results_list.append(fcst_nf)

results_df = pd.concat(results_list)
# results_df = pd.read_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv')

metadata = ['unique_id', 'ds', 'y',
            'trend_str', 'seas_str',
            'is_anomaly', 'large_obs',
            'large_uids', 'anomaly_status', 'Frequency', 'seed']
model_names = results_df.columns[~results_df.columns.str.contains('|'.join(metadata))].tolist()

radar = ModelRadar(cv_df=results_df,
                   metrics=[smape],
                   model_names=model_names,
                   hardness_reference='AutoMLP',
                   ratios_reference='AutoMLP',
                   rope=10)

err = radar.evaluate_by_group(group_col='seed', cv=results_df)
print(err)
err.std(axis=1)

results_df.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=False)
