import pandas as pd
from utilsforecast.losses import smape

from modelradar.evaluate.radar import ModelRadar
from modelradar.visuals.plotter import ModelRadarPlotter, SpiderPlot

from utils.models_config import COLOR_MAPPING

OUTPUT_DIR = 'scripts/experiments/outputs'

cv = pd.read_csv('assets/cv.csv')
cv['anomaly_status'] = cv['is_anomaly'].map({0: 'No anomalies', 1: 'With anomalies'})

monthly_prefix = ['m3_m', 'tourism_m', 'gluonts_m', 'm4_m']
cv['Frequency'] = cv.apply(lambda row: 'Monthly' if any(row['unique_id'].lower().startswith(prefix)
                                                        for prefix in monthly_prefix) else 'Quarterly', axis=1)

metadata = ['unique_id', 'ds', 'y',
            'trend_str', 'seasonal_str',
            'is_anomaly', 'large_obs',
            'large_uids', 'anomaly_status', 'Frequency']
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
cv_hard = radar.cv_df.query(f'unique_id==@radar.uid_accuracy.hard_uid').reset_index(drop=True)

# - overall accuracy
plot2 = radar.evaluate(return_plot=True,
                       # fill_color='#4a5d7c',
                       fill_color=COLOR_MAPPING,
                       flip_coords=True)
plot2.save(f'{OUTPUT_DIR}/plot2_overall.pdf', width=5, height=5)

# - expected shortfall

plot3 = radar.uid_accuracy.expected_shortfall(err,
                                              return_plot=True,
                                              fill_color=COLOR_MAPPING,
                                              flip_coords=True)
plot3.save(f'{OUTPUT_DIR}/plot3_es.pdf', width=5, height=5)

# - winning ratios

plot4 = radar.rope.get_winning_ratios(err, return_plot=True, reference=radar.rope.reference)
plot4.save(f'{OUTPUT_DIR}/plot4_wr.pdf', width=5, height=5)

# - horizon bounds
plot5 = radar.evaluate_by_horizon_bounds(return_plot=True,
                                         plot_model_cats=radar.model_order,
                                         fill_color=COLOR_MAPPING)
plot5.save(f'{OUTPUT_DIR}/plot5_hb.pdf', width=10, height=4)

# - horizon traj

plot6 = radar.evaluate_by_horizon(return_plot=True)
plot6.save(f'{OUTPUT_DIR}/_plot6_horizon.pdf', width=10, height=4)

# - anomalies

plot7 = radar.evaluate_by_group(group_col='anomaly_status',
                                return_plot=True,
                                plot_model_cats=radar.model_order,
                                fill_color=COLOR_MAPPING)
plot7.save(f'{OUTPUT_DIR}/plot7_anomaly.pdf', width=10, height=4)

# - accuracy on hard
plot8 = radar.evaluate(cv=cv_hard,
                       return_plot=True,
                       # fill_color='#4a5d7c',
                       fill_color=COLOR_MAPPING,
                       flip_coords=True)
plot8.save(f'{OUTPUT_DIR}/plot8_accuracy_on_hard.pdf', width=5, height=5)

# - expected shortfall on hard

plot9 = radar.uid_accuracy.expected_shortfall(err_hard,
                                              return_plot=True,
                                              fill_color=COLOR_MAPPING,
                                              flip_coords=True)
plot9.save(f'{OUTPUT_DIR}/plot9_es_hard.pdf', width=5, height=5)

# - s naive dist

plot10 = ModelRadarPlotter.error_histogram(df=err[radar.uid_accuracy.reference].reset_index(),
                                           x_col=radar.uid_accuracy.reference,
                                           x_threshold=radar.uid_accuracy.hardness_threshold,
                                           fill_color='#4a5d7c')

plot10.save(f'{OUTPUT_DIR}/plot10_baseline_dist.pdf', width=10, height=4)

# - freq

plot11 = radar.evaluate_by_group(group_col='Frequency',
                                 return_plot=True,
                                 plot_model_cats=radar.model_order,
                                 fill_color=COLOR_MAPPING)
plot11.save(f'{OUTPUT_DIR}/plot11_freq.pdf', width=10, height=4)

# - winning ratios hard

plot12 = radar.rope.get_winning_ratios(err_hard, return_plot=True, reference=radar.rope.reference)
plot12.save(f'{OUTPUT_DIR}/plot12_wr_hard.pdf', width=5, height=5)

# - trend

plot13 = radar.evaluate_by_group(group_col='trend_str',
                                 return_plot=True,
                                 plot_model_cats=radar.model_order,
                                 fill_color=COLOR_MAPPING)
plot13.save(f'{OUTPUT_DIR}/plot13_trend.pdf', width=10, height=4)

# - seas

plot14 = radar.evaluate_by_group(group_col='seasonal_str',
                                 return_plot=True,
                                 plot_model_cats=radar.model_order,
                                 fill_color=COLOR_MAPPING)
plot14.save(f'{OUTPUT_DIR}/plot14_seas.pdf', width=10, height=4)

# summary

eval_overall = radar.evaluate()
eval_hb = radar.evaluate_by_horizon_bounds()
error_on_anomalies = radar.evaluate_by_group(group_col='anomaly_status')
error_on_freq = radar.evaluate_by_group(group_col='Frequency')
error_on_trend = radar.evaluate_by_group(group_col='trend_str')
error_on_seas = radar.evaluate_by_group(group_col='seasonal_str')
error_on_large_tr = radar.evaluate_by_group(group_col='large_obs')
error_on_large_uid = radar.evaluate_by_group(group_col='large_uids')

df = pd.concat([eval_overall,
                radar.uid_accuracy.expected_shortfall(err),
                eval_hb,
                radar.uid_accuracy.accuracy_on_hard(err),
                error_on_anomalies,
                error_on_trend,
                error_on_seas], axis=1)

TOP_K = 3
top_k_models = set()
for col in df:
    top_on_col = df[col].sort_values().index.tolist()[:TOP_K]
    [top_k_models.add(x) for x in top_on_col]

df = df.loc[list(top_k_models), :]

plot15 = ModelRadarPlotter.multidim_parallel_coords(df, values='normalize')
plot15.save(f'{OUTPUT_DIR}/plot15_all_parallel.pdf', width=10, height=4)

plot16 = SpiderPlot.create_plot(df=df,
                                values='rank',
                                include_title=False,
                                color_set=None)
plot16.save(f'{OUTPUT_DIR}/plot16_all_spider.pdf')
