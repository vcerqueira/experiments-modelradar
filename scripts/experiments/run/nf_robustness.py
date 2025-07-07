# todo when i get some results


SEEDS = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]

# get best params

# for each seed, model
# - run model
# - evaluate

# average results + standard deviation

import warnings

from neuralforecast import NeuralForecast

from utils.load_data.config import DATASETS, DATA_GROUPS
from utils.models_config import ModelsConfig

warnings.filterwarnings('ignore')

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# ---- data loading and partitioning
GROUP_IDX = 1
EXPERIMENT = 'nf'
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=5)
# df = data_loader.get_uid_tails(df, tail_size=100)
# df = data_loader.dummify_series(df)

train, test = data_loader.train_test_split(df, horizon=horizon)

# ---- model setup
nf = NeuralForecast(models=ModelsConfig.get_auto_nf_models(horizon=horizon), freq=freq_str)

# ---- model fitting
nf.fit(df=train)

# ---- forecasts
fcst_nf = nf.predict()
fcst_nf = fcst_nf.reset_index().merge(test, on=['ds', 'unique_id'], how='right')

fcst_nf.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=False)
