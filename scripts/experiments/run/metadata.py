import pandas as pd

from utils.load_data.config import DATASETS, DATA_GROUPS

# ---- data loading and partitioning
EXPERIMENT = 'metadata'

metadata = []
for data_name, group in DATA_GROUPS:
    print(data_name, group)
    data_loader = DATASETS[data_name]

    df, horizon, _, freq_str, _ = data_loader.load_everything(group)

    train, _ = data_loader.train_test_split(df, horizon=horizon)

    n_obs = train.shape[0]
    n_uids = train['unique_id'].nunique()

    metadata.append(
        {
            # 'dataset': data_name,
            # 'group': group,
            'data_group': f'{data_name},{group}',
            'n_obs': n_obs,
            'n_uids': n_uids,
            'Frequency': freq_str,
        }
    )

meta_df = pd.DataFrame(metadata)
meta_df['large_obs'] = (meta_df['n_obs'] > 100_000).map({False: 'Small train', True: 'Large train'})
meta_df['large_uids'] = (meta_df['n_uids'] > 1_000).map({False: 'Low No. TS', True: 'High No. TS'})

meta_df.to_csv(f'assets/results/metadata.csv', index=False)
