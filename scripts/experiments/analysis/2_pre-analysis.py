from pprint import pprint

import pandas as pd
import plotnine as p9
from utilsforecast.losses import smape

from modelradar.evaluate.radar import ModelRadar

from utils.models_config import COLOR_MAPPING

OUTPUT_DIR = 'scripts/experiments/outputs'

cv = pd.read_csv('assets/cv.csv')
cv['anomaly_status'] = cv['is_anomaly'].map({0: 'No anomalies', 1: 'With anomalies'})
cv = cv.drop(columns=['DT', 'RF', 'KNN', 'LGBl', 'Elastic-net'])
cv = cv.rename(columns={
    'Ridge': 'AutoRidge',
    'Lasso': 'AutoLasso',
    'XGB': 'AutoXGBoost',
    'LGB': 'AutoLightGBM',
})

cv['anomaly_status'].value_counts()
cv['anomaly_status'].value_counts(normalize=True)

cv['trend_str'].value_counts(normalize=True)
cv['seas_str'].value_counts(normalize=True)

metadata = ['unique_id', 'ds', 'y',
            'trend_str', 'seas_str',
            'is_anomaly', 'large_obs',
            'large_uids', 'anomaly_status']
model_names = cv.columns[~cv.columns.str.contains('|'.join(metadata))].tolist()

TOP_K = 3

radar = ModelRadar(cv_df=cv,
                   metrics=[smape],
                   model_names=model_names,
                   hardness_reference='SeasonalNaive',
                   ratios_reference='AutoNHITS',
                   rope=10)

err = radar.evaluate(keep_uids=True)
err_hard = radar.uid_accuracy.get_hard_uids(err)

plot1 = radar.evaluate(return_plot=True,
                       # fill_color='#4a5d7c',
                       fill_color=COLOR_MAPPING,
                       flip_coords=False,
                       revert_order=True,
                       extra_theme_settings=p9.theme(plot_margin=0,
                                                     axis_text_x=p9.element_text(angle=60),
                                                     axis_title_y=p9.element_text(size=13),
                                                     axis_text=p9.element_text(size=14,
                                                                               colour='black',
                                                                               weight='bold')
                                                     ))
plot1.save(f'{OUTPUT_DIR}/plot1_preanalysis.pdf', width=11.5, height=5.5)

#

err = radar.evaluate(keep_uids=True)

# multi-dim

eval_overall = radar.evaluate()
eval_hbounds = radar.evaluate_by_horizon_bounds()
error_on_anomalies = radar.evaluate_by_group(group_col='anomaly_status')
error_on_trend = radar.evaluate_by_group(group_col='trend_str')
error_on_seas = radar.evaluate_by_group(group_col='seas_str')
error_on_large_tr = radar.evaluate_by_group(group_col='large_obs')

df = pd.concat([eval_overall,
                radar.uid_accuracy.expected_shortfall(err),
                eval_hbounds,
                radar.uid_accuracy.accuracy_on_hard(err),
                error_on_anomalies,
                error_on_trend,
                error_on_large_tr,
                error_on_seas], axis=1)

top_k_models = set()
for col in df:
    top_on_col = df[col].sort_values().index.tolist()[:TOP_K]
    [top_k_models.add(x) for x in top_on_col]

df = df.loc[list(top_k_models), :]

pprint(df.index.tolist())

SELECTED_MODELS = ['AutoETS', 'AutoNHITS', 'SESOpt',
                   'AutoLightGBM', 'AutoTFT', 'AutoKAN',
                   'AutoARIMA', 'AutoTheta',
                   'AutoDeepNPTS']
