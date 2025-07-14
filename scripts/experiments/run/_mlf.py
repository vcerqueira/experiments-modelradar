"""
Obsolete in final experiments -- keeping for reference
"""
import warnings

from mlforecast import MLForecast
from mlforecast.auto import AutoMLForecast

from utils.load_data.config import DATASETS, DATA_GROUPS
from utils.models_config import ModelsConfig

warnings.filterwarnings("ignore")

# ---- data loading and partitioning
GROUP_IDX = 0
EXPERIMENT = 'mlf'
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

train, test = data_loader.train_test_split(df, horizon=horizon)
# train = data_loader.prune_df_by_size(train, n_lags+horizon+2)
# test = data_loader.prune_df_by_size(test, n_lags+horizon+1)


# ---- model setup
mlf = MLForecast(
    models=ModelsConfig.get_mlf_models(),
    freq=freq_str,
    lags=range(1, n_lags + 1),
)

auto_mlf = AutoMLForecast(
    models=ModelsConfig.get_amlf_models(),
    freq=freq_str,
    season_length=freq_int
)

# ---- model fitting
mlf.fit(df=train)
auto_mlf.fit(df=train,
             n_windows=2,
             refit=False,
             h=horizon,
             num_samples=ModelsConfig.N_SAMPLES)

# ---- forecasts
fcst_mlf = mlf.predict(h=horizon)
fcst_amlf = auto_mlf.predict(h=horizon)

fcst = fcst_mlf.merge(fcst_amlf, on=['ds', 'unique_id'])
fcst = fcst.merge(test, on=['ds', 'unique_id'], how='right')

fcst.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=False)
