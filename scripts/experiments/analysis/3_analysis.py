import pandas as pd
import plotnine as p9
from utilsforecast.losses import smape

from modelradar.evaluate.radar import ModelRadar
from modelradar.visuals.plotter import ModelRadarPlotter, SpiderPlot

OUTPUT_DIR = 'scripts/experiments/outputs'

# add sampling freq dim

cv = pd.read_csv('assets/cv.csv')
cv['anomaly_status'] = cv['is_anomaly'].map({0: 'No anomalies', 1: 'With anomalies'})

metadata = ['unique_id', 'ds', 'y',
            'trend_str', 'seasonal_str',
            'is_anomaly', 'large_obs',
            'large_uids', 'anomaly_status']
model_names = cv.columns[~cv.columns.str.contains('|'.join(metadata))].tolist()

SELECTED_MODELS = ['AutoETS', 'AutoNHITS', 'SESOpt', 'LGB', 'AutoTFT', 'AutoKAN', 'AutoLSTM', 'AutoTheta',
                   'AutoDeepNPTS', 'SeasonalNaive']

radar = ModelRadar(cv_df=cv,
                   metrics=[smape],
                   model_names=SELECTED_MODELS,
                   hardness_reference='SeasonalNaive',
                   ratios_reference='AutoNHITS',
                   rope=10)

err = radar.evaluate(keep_uids=True)
err_hard = radar.uid_accuracy.get_hard_uids(err)

plot1 = radar.evaluate(return_plot=True,
                       fill_color='#4a5d7c',
                       flip_coords=True)
plot1.save(f'{OUTPUT_DIR}/plot2_overall.pdf', width=5, height=5)


radar.rope.get_winning_ratios(err, return_plot=True, reference=radar.rope.reference)
radar.rope.get_winning_ratios(err)

# radar.uid_accuracy.expected_shortfall(err)
# radar.uid_accuracy.expected_shortfall(err, return_plot=True)

eval_overall = radar.evaluate()
eval_overall.sort_values()

# radar.evaluate(cv=cv_hard.reset_index())
eval_hbounds = radar.evaluate_by_horizon_bounds()
eval_hbounds.sort_values('One-step Ahead')
eval_hbounds.sort_values('Multi-step Ahead')

plot = radar.evaluate_by_horizon_bounds(return_plot=True, plot_model_cats=radar.model_order)

eval_fhorizon = radar.evaluate_by_horizon()
plot = radar.evaluate_by_horizon(return_plot=True)

radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='observations')
radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='series')

error_on_anomalies = radar.evaluate_by_group(group_col='anomaly_status')
error_on_trend = radar.evaluate_by_group(group_col='trend_str')
error_on_seas = radar.evaluate_by_group(group_col='seasonal_str')
error_on_large_tr = radar.evaluate_by_group(group_col='large_obs')
error_on_large_uid = radar.evaluate_by_group(group_col='large_uids')

error_on_large_tr.sort_values('Small train')
error_on_large_uid.sort_values('Low No. TS')

# distribution of errors
plot = ModelRadarPlotter.error_distribution(data=err, model_cats=radar.model_order)

# plot.save('test.pdf')


df = pd.concat([eval_overall,
                radar.uid_accuracy.expected_shortfall(err),
                eval_hbounds,
                radar.uid_accuracy.accuracy_on_hard(err),
                error_on_anomalies,
                error_on_trend,
                error_on_seas], axis=1)

k = 2
top_k_models = set()
for col in df:
    top_on_col = df[col].sort_values().index.tolist()[:k]
    [top_k_models.add(x) for x in top_on_col]

df = df.loc[list(top_k_models), :]

plot = ModelRadarPlotter.multidim_parallel_coords(df, values='normalize')
plot.save('test.pdf')

plot = SpiderPlot.create_plot(df=df, values='rank')
plot.save('test.pdf')
