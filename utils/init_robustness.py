import random

import pandas as pd
import numpy as np
from utilsforecast.losses import smape
from modelradar.evaluate.radar import ModelRadar
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

SEEDS = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]


def run_robustness_analysis(best_config, model_name, train_data, test_data, freq_str, horizon):
    """
    Run the best configuration multiple times to test robustness to different seeds
    """
    results = []

    n_runs = len(SEEDS)

    for run_id in range(n_runs):
        print(f"Running {model_name} - Run {run_id + 1}/{n_runs}")
        seed = SEEDS[run_id]
        print('seed:', seed)
        np.random.seed(seed)
        random.seed(seed)

        if model_name == 'LGB':
            model = LGBMRegressor(**best_config['model_params'], seed=seed)
            model_str = LGBMRegressor.__name__
        elif model_name == 'XGB':
            model = XGBRegressor(**best_config['model_params'], seed=seed)
            model_str = XGBRegressor.__name__
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        mlf = MLForecast(
            models=[model],
            freq=freq_str,
            **best_config['mlf_init_params']
        )

        # Fit with model-specific fit parameters
        mlf.fit(train_data, **best_config['mlf_fit_params'])
        preds = mlf.predict(horizon)

        # Add run identifier
        preds['seed'] = seed
        preds = preds.merge(test_data, on=['ds', 'unique_id'], how='right')

        results.append(preds)

    results_df = pd.concat(results, ignore_index=True)
    results_df = results_df.rename(columns={model_str: model_name})

    radar = ModelRadar(cv_df=results_df,
                       metrics=[smape],
                       model_names=[model_name],
                       hardness_reference=model_name,
                       ratios_reference=model_name,
                       rope=10)

    err = radar.evaluate_by_group(group_col='seed', cv=results_df)
    err_s = err.iloc[0]

    return results_df, err_s
