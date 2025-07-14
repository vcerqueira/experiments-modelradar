"""
Obsolete in final experiments -- keeping for reference
"""
import warnings

from neuralforecast import NeuralForecast

from utils.load_data.config import DATASETS, DATA_GROUPS
from utils.models_config import ModelsConfig

warnings.filterwarnings('ignore')

# ---- data loading and partitioning
GROUP_IDX = 0
EXPERIMENT = 'nf-ind0'
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)

train, test = data_loader.train_test_split(df, horizon=horizon)

# ---- model setup
fcst_nf_list = []
test_extended = test.copy()
models = ModelsConfig.get_auto_nf_models(horizon=horizon)
# models = ModelsConfig.get_auto_nf_models_cpu(horizon=horizon)
for mod in models:
    print(mod)
    # mod = ModelsConfig.get_auto_nf_models(horizon=horizon)[0]
    nf = NeuralForecast(models=[mod], freq=freq_str)

    nf.fit(df=train)

    fcst_nf = nf.predict()
    test_extended = test_extended.merge(fcst_nf, on=['ds', 'unique_id'], how='left')

    test_extended.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=False)

    del nf
