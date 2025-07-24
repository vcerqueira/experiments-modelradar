import warnings

from mlforecast.auto import AutoMLForecast
import pandas as pd

from utils.load_data.config import DATASETS, DATA_GROUPS
from utils.models_config import ModelsConfig
from utils.init_robustness import run_robustness_analysis

warnings.filterwarnings("ignore")

# ---- data loading and partitioning
GROUP_IDX = 6
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
             num_samples=ModelsConfig.N_SAMPLES
             # num_samples=2
             )

# ---- forecasts
fcst = auto_mlf.predict(h=horizon)

fcst = fcst.merge(test, on=['ds', 'unique_id'], how='right')

fcst.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=False)

# ---- store best params
best_configs = {}
for mod in auto_mlf.models:
    if mod in ['Ridge', 'Lasso']:
        continue

    best_configs[mod] = auto_mlf.results_[mod].best_trial.user_attrs['config']

# ---- robustness analysis
robustness_results = {}
for model_name, config in best_configs.items():
    print(f"\n=== Testing stability for {model_name} ===")
    print(f"Model-specific mlf_init_params: {config['mlf_init_params']}")
    print(f"Model-specific mlf_fit_params: {config['mlf_fit_params']}")

    robustness_results[model_name], err_s = run_robustness_analysis(
        config, model_name, train, test, freq_str, horizon
    )

robustness_df = pd.merge(robustness_results['XGB'],
                         robustness_results['LGB'].drop(columns=['y']),
                         on=['unique_id', 'ds', 'seed'])

robustness_df.to_csv(f'assets/results/{data_name},{group},robust-{EXPERIMENT}.csv', index=False)
