from typing import Optional
from ast import literal_eval

import pandas as pd
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

from neuralforecast.models import (GRU,
                                   KAN,
                                   MLP,
                                   LSTM,
                                   DLinear,
                                   NHITS,
                                   PatchTST,
                                   TFT,
                                   DeepNPTS,
                                   DeepAR,
                                   TCN,
                                   DilatedRNN)

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

CONFIG_TYPES = {
    'h': int,
    'grid_size': int,
    'spline_order': int,
    'hidden_size': int,
    'learning_rate': float,
    'scaler_type': str,
    'max_steps': int,
    'batch_size': int,
    'windows_batch_size': int,
    'random_seed': int,
    'accelerator': str,
    'input_size': int,
    'limit_val_batches': int,
    'step_size': int,
    'num_layers': int,
    'moving_avg_window': int,
    'n_pool_kernel_size': 'literal_eval',
    'n_freq_downsample': 'literal_eval',
    'dropout': float,
    'n_layers': int,
    'n_head': int,
    'n_heads': int,
    'patch_len': int,
    'revin': bool,
    'encoder_hidden_size': int,
    'encoder_n_layers': int,
    'context_size': int,
    'decoder_hidden_size': int,
    'inference_input_size': int,
    'lstm_hidden_size': int,
    'lstm_n_layers': int,
    'lstm_dropout': float,
    'cell_type': str,
    'dilations': 'literal_eval'
}

SCALER_DEFAULTS = {
    'KAN': 'identity',
    'MLP': 'identity',
    'DLinear': 'identity',
    'NHITS': 'identity',
    'DeepNPTS': 'identity',
    'TFT': 'robust',
    'PatchTST': 'identity',
    'GRU': 'robust',
    'DeepAR': 'identity',
    'LSTM': 'robust',
    'DilatedRNN': 'robust',
    'TCN': 'robust',
}

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

    @classmethod
    def get_sf_models(cls, season_len: int, input_size: int):
        models = [
            RandomWalkWithDrift(),
            SeasonalNaive(season_length=season_len),
            AutoETS(season_length=season_len),
            AutoARIMA(season_length=season_len,
                      max_p=2, max_q=2,
                      max_P=1, max_Q=1, max_d=1,
                      max_D=1, nmodels=cls.N_SAMPLES),
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

    @staticmethod
    def convert_config_types(d: dict) -> dict:
        """Convert dictionary values based on CONFIG_TYPES mapping.

        Args:
            d: Dictionary with keys matching CONFIG_TYPES

        Returns:
            Dictionary with values converted to specified types

        Raises:
            KeyError: If d contains keys not in CONFIG_TYPES
        """
        unknown_keys = set(d.keys()) - set(CONFIG_TYPES.keys())
        if unknown_keys:
            raise KeyError(f"Unknown config keys: {unknown_keys}")

        converted = {}
        for k, v in d.items():
            type_converter = CONFIG_TYPES[k]
            if type_converter == 'literal_eval':
                converted[k] = literal_eval(v)
            else:
                converted[k] = type_converter(v)
        return converted

    @classmethod
    def get_nf_models_config(
            cls,
            horizon: int,
            limit_val_batches: bool = False,
            best_configs: Optional[pd.DataFrame] = None,
            random_seed: Optional[int] = None):

        NEED_CPU = ['GRU',
                    'DeepNPTS',
                    'DLinear',
                    'DeepAR',
                    'LSTM',
                    'KAN',
                    'DilatedRNN',
                    'TCN']

        model_cls = {
            'KAN': KAN,
            'MLP': MLP,
            'DLinear': DLinear,
            'NHITS': NHITS,
            'DeepNPTS': DeepNPTS,
            'TFT': TFT,
            'PatchTST': PatchTST,
            'GRU': GRU,
            'DeepAR': DeepAR,
            'LSTM': LSTM,
            'DilatedRNN': DilatedRNN,
            'TCN': TCN,
        }

        models = []
        for mod_name, mod in model_cls.items():
            # mod_name = 'AutoKAN'
            # mod_name = 'KAN'
            conf = best_configs[f'Auto{mod_name}'].copy()
            conf['max_steps'] = int(float(conf['max_steps']))
            conf.pop('loss')
            conf.pop('valid_loss')
            conf.pop('h')

            if pd.isna(conf['scaler_type']):
                # conf['scaler_type'] = 'identity'
                conf['scaler_type'] = SCALER_DEFAULTS[mod_name]

            conf = conf.dropna().to_dict()

            conf = cls.convert_config_types(conf)
            print(mod_name, conf)

            if random_seed is not None:
                conf['random_seed'] = random_seed

            if mod_name in NEED_CPU:
                # for RNN's
                conf['accelerator'] = 'cpu'
            else:
                conf['accelerator'] = 'mps'

            if limit_val_batches:
                # for M4
                conf['limit_val_batches'] = 50

            model_instance = mod(
                h=horizon,
                alias=f'Auto{mod_name}',
                **conf
            )

            models.append(model_instance)

        return models
