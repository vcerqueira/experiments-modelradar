import pandas as pd
import plotnine as p9
from utilsforecast.losses import smape

from modelradar.evaluate.radar import ModelRadar
from modelradar.visuals.plotter import ModelRadarPlotter, SpiderPlot

from utils.models_config import COLOR_MAPPING

OUTPUT_DIR = 'scripts/experiments/outputs'

cv = pd.read_csv('assets/cv.csv')
cv['anomaly_status'] = cv['is_anomaly'].map({0: 'Non-anomalies', 1: 'Anomalies'})
cv = cv.rename(columns={
    'Ridge': 'AutoRidge',
    'Lasso': 'AutoLasso',
    'XGB': 'AutoXGBoost',
    'LGB': 'AutoLightGBM',
})

# cv['anomaly_status'].value_counts()
# cv['anomaly_status'].value_counts(normalize=True)

monthly_prefix = ['m3_m', 'tourism_m', 'gluonts_m', 'm4_m']
cv['Frequency'] = cv.apply(lambda row: 'Monthly' if any(row['unique_id'].lower().startswith(prefix)
                                                        for prefix in monthly_prefix) else 'Quarterly', axis=1)

metadata = ['unique_id', 'ds', 'y', 'index',
            'trend_str', 'seas_str',
            'is_anomaly', 'large_obs', 'data_group',
            'large_uids', 'anomaly_status', 'Frequency']
model_names = cv.columns[~cv.columns.str.contains('|'.join(metadata))].tolist()

SELECTED_MODELS = ['AutoLightGBM',
                   'AutoETS',
                   'AutoARIMA',
                   'AutoPatchTST',
                   'AutoDilatedRNN',
                   'AutoNHITS',
                   'AutoTheta',
                   'AutoMLP',
                   'AutoTFT',
                   'SeasonalNaive',
                   'SESOpt']

radar = ModelRadar(cv_df=cv,
                   metrics=[smape],
                   model_names=SELECTED_MODELS,
                   hardness_reference='SeasonalNaive',
                   ratios_reference='AutoPatchTST',
                   rope=10)

print(radar.evaluate())
err = radar.evaluate(keep_uids=True)
err_hard = radar.uid_accuracy.get_hard_uids(err)
cv_hard = radar.cv_df.query(f'unique_id==@radar.uid_accuracy.hard_uid').reset_index(drop=True)

# - overall accuracy
plot2 = radar.evaluate(return_plot=True,
                       fill_color=COLOR_MAPPING,
                       flip_coords=True,
                       extra_theme_settings=p9.theme(plot_margin=0,
                                                     axis_text=p9.element_text(size=15,
                                                                               colour='black',
                                                                               weight='bold'),
                                                     axis_title_x=p9.element_blank()), )
plot2.save(f'{OUTPUT_DIR}/plot2_overall.pdf', width=6, height=5)

# - expected shortfall

plot3 = radar.uid_accuracy.expected_shortfall(err,
                                              return_plot=True,
                                              fill_color=COLOR_MAPPING,
                                              flip_coords=True,
                                              extra_theme_settings=p9.theme(plot_margin=0,
                                                                            axis_text=p9.element_text(size=15,
                                                                                                      colour='black',
                                                                                                      weight='bold'),
                                                                            axis_title_x=p9.element_blank()))
plot3.save(f'{OUTPUT_DIR}/plot3_es.pdf', width=6, height=5)

# - winning ratios

plot4 = radar.rope.get_winning_ratios(err,
                                      return_plot=True,
                                      reference=radar.rope.reference,
                                      extra_theme_settings=p9.theme(plot_margin=0,
                                                                    legend_text=p9.element_text(size=11),
                                                                    axis_text=p9.element_text(size=15,
                                                                                              colour='black',
                                                                                              weight='bold'),
                                                                    axis_title_x=p9.element_blank()))
plot4.save(f'{OUTPUT_DIR}/plot4_wr.pdf', width=6, height=5)

# - winning ratios hard

plot12 = radar.rope.get_winning_ratios(err_hard,
                                       return_plot=True,
                                       reference=radar.rope.reference,
                                       extra_theme_settings=p9.theme(plot_margin=0,
                                                                     legend_text=p9.element_text(size=11),
                                                                     axis_text=p9.element_text(size=15,
                                                                                               colour='black',
                                                                                               weight='bold'),
                                                                     axis_title_x=p9.element_blank()))
plot12.save(f'{OUTPUT_DIR}/plot12_wr_hard.pdf', width=6, height=5)

# - winning ratios

radar.rope.rope = 0

plot4 = radar.rope.get_winning_ratios(err,
                                      return_plot=True,
                                      reference=radar.rope.reference,
                                      extra_theme_settings=p9.theme(plot_margin=0,
                                                                    legend_text=p9.element_text(size=11),
                                                                    axis_text=p9.element_text(size=15,
                                                                                              colour='black',
                                                                                              weight='bold'),
                                                                    axis_title_x=p9.element_blank()))
plot4.save(f'{OUTPUT_DIR}/plot4_wr_rope0.pdf', width=6, height=5)

# - winning ratios hard

plot12 = radar.rope.get_winning_ratios(err_hard,
                                       return_plot=True,
                                       reference=radar.rope.reference,
                                       extra_theme_settings=p9.theme(plot_margin=0,
                                                                     legend_text=p9.element_text(size=11),
                                                                     axis_text=p9.element_text(size=15,
                                                                                               colour='black',
                                                                                               weight='bold'),
                                                                     axis_title_x=p9.element_blank()))
plot12.save(f'{OUTPUT_DIR}/plot12_wr_hard_rope0.pdf', width=6, height=5)

# - horizon bounds
plot5 = radar.evaluate_by_horizon_bounds(return_plot=True,
                                         plot_model_cats=radar.model_order,
                                         fill_color=COLOR_MAPPING,
                                         extra_theme_settings=p9.theme(plot_margin=0,
                                                                       strip_background_x=p9.element_text(
                                                                           colour='lightgrey'),
                                                                       strip_text=p9.element_text(size=18),
                                                                       axis_text_x=p9.element_text(angle=60),
                                                                       axis_title_y=p9.element_text(size=15),
                                                                       axis_text=p9.element_text(size=15,
                                                                                                 colour='black',
                                                                                                 weight='bold'),
                                                                       ))
plot5.save(f'{OUTPUT_DIR}/plot5_hb.pdf', width=11, height=5.5)

# - horizon traj

# plot6 = radar.evaluate_by_horizon(return_plot=True)
# plot6.save(f'{OUTPUT_DIR}/_plot6_horizon.pdf', width=10, height=4)

# - anomalies

plot7 = radar.evaluate_by_group(group_col='anomaly_status',
                                return_plot=True,
                                plot_model_cats=radar.model_order,
                                fill_color=COLOR_MAPPING,
                                extra_theme_settings=p9.theme(plot_margin=0,
                                                              strip_text=p9.element_text(size=18),
                                                              axis_title_y=p9.element_text(size=15),
                                                              axis_text_x=p9.element_text(angle=60),
                                                              axis_text=p9.element_text(size=15,
                                                                                        colour='black',
                                                                                        weight='bold')
                                                              ))
plot7.save(f'{OUTPUT_DIR}/plot7_anomaly.pdf', width=12, height=5.5)

# - accuracy on hard
plot8 = radar.evaluate(cv=cv_hard,
                       return_plot=True,
                       fill_color=COLOR_MAPPING,
                       flip_coords=True,
                       extra_theme_settings=p9.theme(plot_margin=0,
                                                     axis_text=p9.element_text(size=15,
                                                                               colour='black',
                                                                               weight='bold'),
                                                     axis_title_x=p9.element_blank()))
plot8.save(f'{OUTPUT_DIR}/plot8_accuracy_on_hard.pdf', width=6, height=5)

# - expected shortfall on hard

plot9 = radar.uid_accuracy.expected_shortfall(err_hard,
                                              return_plot=True,
                                              fill_color=COLOR_MAPPING,
                                              flip_coords=True,
                                              extra_theme_settings=p9.theme(plot_margin=0,
                                                                            axis_text=p9.element_text(size=15,
                                                                                                      colour='black',
                                                                                                      weight='bold'),
                                                                            axis_title_x=p9.element_blank()))
plot9.save(f'{OUTPUT_DIR}/plot9_es_hard.pdf', width=6, height=5)

# - s naive dist

plot10 = ModelRadarPlotter.error_histogram(df=err[radar.uid_accuracy.reference].reset_index(),
                                           x_col=radar.uid_accuracy.reference,
                                           x_threshold=radar.uid_accuracy.hardness_threshold,
                                           fill_color='#4a5d7c')
plot10 = plot10 + p9.theme(plot_margin=0.025,
                           axis_text=p9.element_text(size=14),
                           axis_title=p9.element_text(size=14))

plot10.save(f'{OUTPUT_DIR}/plot10_baseline_dist.pdf', width=11, height=5.5)

# - freq

plot11 = radar.evaluate_by_group(group_col='Frequency',
                                 return_plot=True,
                                 plot_model_cats=radar.model_order,
                                 fill_color=COLOR_MAPPING,
                                 extra_theme_settings=p9.theme(plot_margin=0,
                                                               strip_text=p9.element_text(size=18),
                                                               axis_text_x=p9.element_text(size=15,
                                                                                           angle=60,
                                                                                           colour='black',
                                                                                           weight='bold'),
                                                               axis_title_x=p9.element_blank()))
plot11.save(f'{OUTPUT_DIR}/plot11_freq.pdf', width=11, height=5.5)

# - trend

plot13 = radar.evaluate_by_group(group_col='trend_str',
                                 return_plot=True,
                                 plot_model_cats=radar.model_order,
                                 fill_color=COLOR_MAPPING,
                                 extra_theme_settings=p9.theme(plot_margin=0,
                                                               strip_text=p9.element_text(size=18),
                                                               axis_title_y=p9.element_text(size=15),
                                                               axis_text_y=p9.element_text(size=15),
                                                               axis_text_x=p9.element_text(size=15,
                                                                                           angle=60,
                                                                                           colour='black',
                                                                                           weight='bold'),
                                                               axis_title_x=p9.element_blank()))
plot13.save(f'{OUTPUT_DIR}/plot13_trend.pdf', width=11, height=5.5)

# - seas

plot14 = radar.evaluate_by_group(group_col='seas_str',
                                 return_plot=True,
                                 plot_model_cats=radar.model_order,
                                 fill_color=COLOR_MAPPING,
                                 extra_theme_settings=p9.theme(plot_margin=0,
                                                               strip_text=p9.element_text(size=18),
                                                               axis_title_y=p9.element_text(size=15),
                                                               axis_text_y=p9.element_text(size=15),
                                                               axis_text_x=p9.element_text(size=15,
                                                                                           angle=60,
                                                                                           colour='black',
                                                                                           weight='bold'),
                                                               axis_title_x=p9.element_blank()))
plot14.save(f'{OUTPUT_DIR}/plot14_seas.pdf', width=11, height=5.5)

# summary

eval_overall = radar.evaluate()
eval_hb = radar.evaluate_by_horizon_bounds()
error_on_anomalies = radar.evaluate_by_group(group_col='anomaly_status')
error_on_freq = radar.evaluate_by_group(group_col='Frequency')
error_on_trend = radar.evaluate_by_group(group_col='trend_str')
error_on_seas = radar.evaluate_by_group(group_col='seas_str')
error_on_large_tr = radar.evaluate_by_group(group_col='large_obs')
error_on_large_uid = radar.evaluate_by_group(group_col='large_uids')
eval_on_hard = radar.uid_accuracy.accuracy_on_hard(err)
eval_on_hard.name = 'Hard'

df = pd.concat([eval_overall,
                radar.uid_accuracy.expected_shortfall(err),
                eval_hb,
                eval_on_hard,
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
plot16 = plot16 + p9.theme(plot_margin=0.05,
                           legend_position='top',
                           legend_text=p9.element_text(size=17),
                           legend_key_size=20,
                           legend_key_width=20)
plot16.save(f'{OUTPUT_DIR}/plot16_all_spider.pdf', width=12, height=16)
