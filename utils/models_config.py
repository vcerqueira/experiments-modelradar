from neuralforecast.auto import (AutoGRU,
                                 AutoKAN,
                                 AutoMLP,
                                 AutoLSTM,
                                 AutoDLinear,
                                 AutoNHITS,
                                 AutoPatchTST,
                                 AutoTFT,
                                 AutoDeepNPTS,
                                 AutoDeepAR,
                                 AutoTCN,
                                 AutoDilatedRNN)

import lightgbm as lgb
import xgboost as xgb
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

COLORS = {
    'A': '#8B0000',  # Deep red
    'B': '#6A0DAD',  # Deep purple
    'C': '#20B2AA'  # Light sea green
}

COLOR_MAPPING = {'AutoTFT': COLORS['A'],
                 'AutoPatchTST': COLORS['A'],
                 'AutoKAN': COLORS['A'],
                 'AutoMLP': COLORS['A'],
                 'AutoGRU': COLORS['A'],
                 'AutoLSTM': COLORS['A'],
                 'AutoDLinear': COLORS['A'],
                 'AutoDeepAR': COLORS['A'],
                 'AutoDeepAR-median': COLORS['A'],
                 'AutoNHITS': COLORS['A'],
                 'AutoDeepNPTS': COLORS['A'],
                 'AutoTCN': COLORS['A'],
                 'AutoDilatedRNN': COLORS['A'],
                 'RWD': COLORS['B'],
                 'SeasonalNaive': COLORS['B'],
                 'AutoETS': COLORS['B'],
                 'AutoARIMA': COLORS['B'],
                 'AutoTheta': COLORS['B'],
                 'SESOpt': COLORS['B'],
                 'CrostonOptimized': COLORS['B'],
                 'WindowAverage': COLORS['B'],
                 'Ridge': COLORS['C'],
                 'Lasso': COLORS['C'],
                 'XGB': COLORS['C'],
                 'LGB': COLORS['C'],
                 }


class ModelsConfig:
    N_SAMPLES = 100

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
            # 'DT': DecisionTreeRegressor(max_depth=5),
            # 'RF': xgb.XGBRFRegressor(n_estimators=25),
            # 'KNN': KNeighborsRegressor(n_neighbors=50),
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
    def get_auto_nf_models(cls, horizon, limit_val_batches: bool = False):
        NEED_CPU = ['AutoGRU',
                    'AutoDeepNPTS',
                    'AutoDeepAR',
                    'AutoLSTM',
                    'AutoKAN',
                    'AutoDilatedRNN',
                    'AutoTCN']

        model_cls = {
            'AutoKAN': AutoKAN,
            'AutoMLP': AutoMLP,
            'AutoDLinear': AutoDLinear,
            'AutoNHITS': AutoNHITS,
            'AutoDeepNPTS': AutoDeepNPTS,
            'AutoTFT': AutoTFT,
            'AutoPatchTST': AutoPatchTST,
            'AutoGRU': AutoGRU,
            'AutoDeepAR': AutoDeepAR,
            'AutoLSTM': AutoLSTM,
            'AutoDilatedRNN': AutoDilatedRNN,
            'AutoTCN': AutoTCN,
        }

        models = []
        for mod_name, mod in model_cls.items():
            if mod_name in NEED_CPU:
                # for RNN's
                mod.default_config['accelerator'] = 'cpu'
            else:
                mod.default_config['accelerator'] = 'mps'

            if limit_val_batches:
                # for M4
                mod.default_config['limit_val_batches'] = 50

            model_instance = mod(
                h=horizon,
                num_samples=cls.N_SAMPLES,
                alias=mod_name,
            )

            models.append(model_instance)

        return models

    @classmethod
    def get_auto_nf_models_simple(cls, horizon):
        models = [
            AutoKAN(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoKAN'),
            AutoMLP(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoMLP'),
            AutoDLinear(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoDLinear'),
            AutoNHITS(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoNHITS'),
            AutoDeepNPTS(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoDeepNPTS'),
            AutoTFT(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoTFT'),
            AutoPatchTST(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoPatchTST'),
            AutoGRU(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoGRU'),
            AutoDeepAR(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoDeepAR'),
            AutoLSTM(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoLSTM'),
            AutoDilatedRNN(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoDilatedRNN'),
            AutoTCN(h=horizon, num_samples=cls.N_SAMPLES, alias='AutoTCN'),
        ]
        return models

    @classmethod
    def get_auto_nf_models_cpu(cls, horizon):
        model_cls = {
            'AutoKAN': AutoKAN,
            'AutoMLP': AutoMLP,
            'AutoDLinear': AutoDLinear,
            'AutoNHITS': AutoNHITS,
            'AutoDeepNPTS': AutoDeepNPTS,
            'AutoTFT': AutoTFT,
            'AutoGRU': AutoGRU,
            'AutoLSTM': AutoLSTM,
            'AutoDeepAR': AutoDeepAR,
            'AutoDilatedRNN': AutoDilatedRNN,
            'AutoTCN': AutoTCN,
            'AutoPatchTST': AutoPatchTST,
        }

        models = []
        for mod_name, mod in model_cls.items():
            # for RNN's
            mod.default_config['accelerator'] = 'cpu'
            # for M4
            # mod.default_config['limit_val_batches'] = 50

            model_instance = mod(
                h=horizon,
                num_samples=cls.N_SAMPLES,
                alias=mod_name,
            )

            models.append(model_instance)

        return models
