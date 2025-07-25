import warnings

import pandas as pd
import plotnine as p9
from utilsforecast.losses import smape
from modelradar.evaluate.radar import ModelRadar

from utils.load_data.config import DATA_GROUPS

warnings.filterwarnings('ignore')
OUTPUT_DIR = 'scripts/experiments/outputs'

results_list = []
for data_name, group in DATA_GROUPS:
    # data_name = 'Gluonts'
    # group = 'm1_monthly'
    print(data_name, group)
    # df = pd.read_csv(f'assets/results/{data_name},{group},robust-nf.csv')
    df_nf = pd.read_csv(f'assets/results/{data_name},{group},robust-nf.csv')
    df_mlf = pd.read_csv(f'assets/results/{data_name},{group},robust-mlf.csv')
    df_mlf = df_mlf.rename(columns={'XGB': 'AutoXGBoost', 'LGB': 'AutoLightGBM'})

    df = pd.merge(df_nf,df_mlf.drop(columns=['y']), on=['unique_id', 'ds', 'seed'])

    # df = df_nf.merge(df_mlf)

    df['Dataset'] = f'{data_name},{group}'

    results_list.append(df)

results_df = pd.concat(results_list)

metadata = ['unique_id', 'ds', 'y', '-hi', '-lo',
            'trend_str', 'seas_str', 'Dataset',
            'is_anomaly', 'large_obs',
            'large_uids', 'anomaly_status', 'Frequency', 'seed']
model_names = results_df.columns[~results_df.columns.str.contains('|'.join(metadata))].tolist()

radar = ModelRadar(cv_df=results_df,
                   metrics=[smape],
                   model_names=model_names,
                   hardness_reference='AutoMLP',
                   ratios_reference='AutoMLP',
                   rope=10)

err_df_ = radar.evaluate_by_group(group_col='seed')

err_melted = err_df_.T.reset_index().melt('index')

data_melted = err_melted.rename(columns={"variable": "Model"})

data_melted["Model"] = pd.Categorical(
    data_melted["Model"].values.tolist(), categories=data_melted["Model"].sort_values().unique()
)

plot17 = (
        p9.ggplot(
            data_melted,
            p9.aes(
                x="Model",
                y="value",
            ),
        )
        + p9.theme_538(base_family="Palatino", base_size=12) +
        p9.theme(
            plot_margin=0.025,
            panel_background=p9.element_rect(fill="white"),
            plot_background=p9.element_rect(fill="white"),
            legend_box_background=p9.element_rect(fill="white"),
            strip_background=p9.element_rect(fill="white"),
            legend_background=p9.element_rect(fill="white"),
            axis_text_x=p9.element_text(size=13, angle=30),
            axis_text_y=p9.element_text(size=13),
            legend_title=p9.element_blank(),
        ) + p9.geom_boxplot(width=0.8, show_legend=False)
        + p9.labs(y="Error", x="")
        + p9.guides(fill="none")
)

plot17.save(f'{OUTPUT_DIR}/plot17_robustness.pdf', width=11, height=5.5)
