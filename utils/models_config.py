from neuralforecast.auto import (AutoGRU,
                                 AutoKAN,
                                 AutoMLP,
                                 AutoLSTM,
                                 AutoDLinear,
                                 AutoNHITS,
                                 AutoAutoformer,
                                 AutoInformer,
                                 AutoDeepNPTS,
                                 AutoDeepAR,
                                 AutoTCN,
                                 AutoDilatedRNN)

import lightgbm as lgb
import xgboost as xgb
from ray import tune
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from mlforecast.auto import (AutoLasso,
                             AutoRidge,
                             AutoXGBoost,
                             AutoLightGBM,
                             AutoElasticNet)

from statsforecast.models import (
    SeasonalNaive,
    AutoETS,
    AutoARIMA,
    RandomWalkWithDrift,
    AutoTheta,
    SimpleExponentialSmoothingOptimized,
    CrostonOptimized,
    WindowAverage
)


class ModelsConfig:
    N_SAMPLES = 20

    @staticmethod
    def get_sf_models(season_len: int, input_size: int):
        models = [
            RandomWalkWithDrift(),
            SeasonalNaive(season_length=season_len),
            AutoETS(season_length=season_len),
            AutoARIMA(season_length=season_len, max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1, nmodels=5),
            AutoTheta(season_length=season_len),
            SimpleExponentialSmoothingOptimized(),
            CrostonOptimized(),
            WindowAverage(window_size=input_size),
        ]

        return models

    @staticmethod
    def get_sf_anomaly_model(season_len: int):
        models = [
            SeasonalNaive(season_length=season_len),
        ]

        return models

    @staticmethod
    def get_mlf_models():
        models_ml = {
            'DT': DecisionTreeRegressor(max_depth=5),
            'RF': xgb.XGBRFRegressor(n_estimators=25),
            'KNN': KNeighborsRegressor(n_neighbors=50),
            'LGBl': lgb.LGBMRegressor(verbosity=-1, n_jobs=1, linear_tree=True),
        }

        return models_ml

    @staticmethod
    def get_amlf_models():
        auto_models_ml = {
            'Ridge': AutoRidge(),
            'Lasso': AutoLasso(),
            'Elastic-net': AutoElasticNet(),
            'XGB': AutoXGBoost(),
            'LGB': AutoLightGBM(),
        }

        return auto_models_ml

    @classmethod
    def get_auto_nf_models(cls, horizon):
        models = [
            AutoKAN(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoKAN'),
            AutoMLP(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoMLP'),
            AutoGRU(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoGRU'),
            AutoLSTM(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoLSTM'),
            AutoDLinear(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoDLinear'),
            AutoDeepAR(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoDeepAR'),
            AutoNHITS(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoNHITS'),
            AutoDeepNPTS(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoDeepNPTS'),
            AutoAutoformer(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoAutoformer'),
            AutoInformer(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoInformer'),
            AutoTCN(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoTCN'),
            AutoDilatedRNN(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoDilatedRNN'),
        ]
        return models

    @classmethod
    def get_auto_nf_models_no_robust(cls, horizon):
        model_cls = [
            AutoKAN,
            AutoMLP,
            AutoGRU,
            AutoLSTM,
            AutoDLinear,
            AutoDeepAR,
            AutoNHITS,
            AutoDeepNPTS,
            AutoAutoformer,
            AutoInformer,
            AutoTCN,
            AutoDilatedRNN,
        ]

        models = []
        for mod in model_cls:
            if 'scaler_type' in mod.default_config:
                current_choices = mod.default_config['scaler_type'].categories
                new_choices = [x for x in current_choices if x != 'robust']
                mod.default_config['scaler_type'] = tune.choice(new_choices)

                # from pprint import pprint
                # pprint(mod.default_config)

            model_instance = mod(
                h=horizon,
                num_samples=cls.N_SAMPLES
            )

            models.append(model_instance)

        return models
