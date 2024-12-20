import warnings

from statsforecast import StatsForecast

from utils.load_data.config import DATASETS, DATA_GROUPS, GROUP_IDX
from utils.models_config import ModelsConfig

warnings.filterwarnings("ignore")

# ---- data loading and partitioning
EXPERIMENT = 'sfa'
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=5)
df = data_loader.get_uid_tails(df, tail_size=100)
df = data_loader.dummify_series(df)

train, test = data_loader.train_test_split(df, horizon=horizon)

# ---- model setup
sf = StatsForecast(
    models=ModelsConfig.get_sf_anomaly_model(season_len=freq_int),
    freq=freq_str,
    n_jobs=1)

# ---- model fitting
sf.fit(df=train)

# ---- forecasts
fcst_sf = sf.predict(h=horizon, level=[99])
fcst_sf = fcst_sf.reset_index().merge(test, on=['ds', 'unique_id'], how='right')

is_outside_pi = (fcst_sf['y'] >= fcst_sf['SeasonalNaive-hi-99']) | (fcst_sf['y'] <= fcst_sf['SeasonalNaive-lo-99'])
is_outside_pi = is_outside_pi.astype(int)
is_anomaly_int = is_outside_pi.astype(int)

fcst_sf['is_anomaly'] = is_anomaly_int
fcst_sf = fcst_sf.drop(columns=['SeasonalNaive-hi-99','SeasonalNaive','SeasonalNaive-lo-99'])

fcst_sf.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=False)
