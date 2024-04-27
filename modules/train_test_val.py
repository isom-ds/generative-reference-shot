import pandas as pd
from sklearn.utils import shuffle

def load_data(
    dir: str,
    split: list = [60, 20, 20],
    seed: int = 12345
    ):

    final_output = {}

    for dataset in ['affectivetext', 'crowdflower', 'electoraltweets', 'ssec', 'emoint', 'goemotions', 'tec']:
        dataset_output = {}

        # Load Data
        df_original = pd.read_parquet(f'{dir}//original//{dataset}.parquet')
        df_primary = pd.read_parquet(f'{dir}//primary//{dataset}.parquet')
        df_secondary = pd.read_parquet(f'{dir}//secondary//{dataset}.parquet')

        def splits(data):
            # Output dict
            output = {}

            # Randomise
            df = shuffle(data, random_state=seed).reset_index(drop=True)

            # Split datasets - original split
            if len(df['dataset'].unique()) != 1:
                if len(df['dataset'].unique()) == 3:
                    train = df.loc[df['dataset'] == 'train', :]
                    test = df.loc[df['dataset'] == 'test', :]
                    val = df.loc[df['dataset'] == 'validation', :]
                else:
                    train = df.loc[df['dataset'] == 'train', :]
                    test = df.loc[df['dataset'] == 'test', :]
                    val = df.loc[df['dataset'] == 'test', :]

                train = train.assign(dataset = 'train')
                test = test.assign(dataset = 'test')
                val = val.assign(dataset = 'val')

                print(f"|---- DEFAULT SPLIT = Train: {len(train)} | Test: {len(test)} | Validation: {len(val)}")

                output['default'] = pd.concat([train, test, val])

            # Split datasets - custom split
            splits = [int(round(sum(split[:i+1])/100 * len(df), 0)) for i,n in enumerate(split)]
            if len(df) % 3:
                splits[0] += 1
            train = df.loc[:splits[0]+1, :]
            test = df.loc[splits[0]:splits[1]+1, :]
            val = df.loc[splits[1]:, :]

            train = train.assign(dataset = 'train')
            test = test.assign(dataset = 'test')
            val = val.assign(dataset = 'val')

            print(f"|---- {'-'.join([str(i) for i in split])} SPLIT = Train:{len(train)} | Test: {len(test)} | Validation: {len(val)}")

            output['custom_split'] = pd.concat([train, test, val])

            return output

        print(f"{dataset}: {len(df_original)}")
        print(f"|-- ORIGINAL: {len(df_original)}")
        dataset_output['original'] = splits(df_original)

        print(f"|-- PRIMARY: {len(df_primary)}")
        dataset_output['primary'] = splits(df_primary)

        print(f"|-- SECONDARY: {len(df_secondary)}")
        dataset_output['secondary'] = splits(df_secondary)
        print("")

        final_output[dataset] = dataset_output

    return final_output