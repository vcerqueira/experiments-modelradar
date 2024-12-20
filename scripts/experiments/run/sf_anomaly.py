import warnings

import pandas as pd
from statsforecast import StatsForecast

from utils.load_data.config import DATASETS
from utils.models_config import ModelsConfig


warnings.filterwarnings("ignore")

# ---- data loading and partitioning
data_name, group = 'Gluonts', 'm1_monthly'
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=5)
df = data_loader.get_uid_tails(df, tail_size=100)
df = data_loader.dummify_series(df)


train, test = data_loader.train_test_split(df, horizon=horizon)

# ---- model setup
print('...Model setup')

sf = StatsForecast(
    models=ModelsConfig.get_sf_models(season_len=freq_int, input_size=n_lags),
    freq=freq_str,
    n_jobs=1,
)

# ---- model fitting
sf.fit(df=train)
# sf.forecast(fitted=True, h=1)

# ---- forecasts
print('...getting predictions')

# fcst_insample_sf = sf.forecast_fitted_values()
fcst_sf = sf.predict(h=horizon)