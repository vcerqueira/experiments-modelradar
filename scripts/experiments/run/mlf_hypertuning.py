import warnings

import pandas as pd
from mlforecast.auto import AutoMLForecast

from utils.load_data.config import DATASETS, DATA_GROUPS
from utils.models_config import ModelsConfig

warnings.filterwarnings("ignore")

# ---- data loading and partitioning
GROUP_IDX = 0
EXPERIMENT = 'hpo-mlf'
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df = data_loader.prune_df_by_size(df, (n_lags+horizon+1))
df = data_loader.prune_df_by_size(df, (n_lags + horizon) * 2 + 1)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=5)
# df = data_loader.get_uid_tails(df, tail_size=100)
# df = data_loader.dummify_series(df)
# df['unique_id'].value_counts()

train, _ = data_loader.train_test_split(df, horizon=horizon)
# train = data_loader.prune_df_by_size(train, n_lags+horizon+2)
# test = data_loader.prune_df_by_size(test, n_lags+horizon+1)

auto_mlf = AutoMLForecast(
    models=ModelsConfig.get_amlf_models(),
    freq=freq_str,
    season_length=freq_int
)

# ---- model fitting
auto_mlf.fit(df=train,
             n_windows=2,
             refit=False,
             h=horizon,
             num_samples=ModelsConfig.N_SAMPLES)

# ---- store best params
best_configs = {}
for mod in auto_mlf.models:
    best_configs[mod] = auto_mlf.results_[mod].best_trial.user_attrs['config']

best_conf_df = pd.DataFrame(best_configs)
best_conf_df.index.name = 'parameter'

best_conf_df.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=True)
