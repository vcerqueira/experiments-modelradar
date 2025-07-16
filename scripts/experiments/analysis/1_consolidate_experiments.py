import pandas as pd

from utils.load_data.config import DATA_GROUPS

VARIANTS = ['nf', 'sf', 'mlf']
RESULTS_DIR = 'assets/results'

metadata = pd.read_csv(f'{RESULTS_DIR}/metadata.csv')

dataset = []
for ds, group in DATA_GROUPS:

    df = pd.read_csv(f'{RESULTS_DIR}/{ds},{group},sfa.csv')
    for variant in VARIANTS:
        file = f'{RESULTS_DIR}/{ds},{group},{variant}.csv'

        df_var = pd.read_csv(file)
        df = df.merge(df_var.drop(columns='y'), on=['unique_id', 'ds'])

    df = df.drop(columns=df.columns[df.columns.str.contains('-hi-|-lo-')])
    feats = pd.read_csv(f'{RESULTS_DIR}/{ds},{group},features.csv')

    df = df.merge(feats, on='unique_id', how='left')

    df['unique_id'] = ds + '_' + df['unique_id'].astype(str)
    df['data_group'] = f'{ds},{group}'

    dataset.append(df)

df_all = pd.concat(dataset)

df_all = df_all.merge(metadata, on='data_group', how='left')
df_all = df_all.drop(columns=['n_obs','n_uids'])

df_all.to_csv('assets/cv.csv', index=False)
