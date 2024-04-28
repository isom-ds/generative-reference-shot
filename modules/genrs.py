#%%
import pandas as pd
from roberta_pipeline import *

#%%
def GenRS_sample(
    reference,
    target,
    seed = 12345
    ):

    # Copy datasets
    df_reference = reference.copy()
    df_target = target.copy()

    # Create output dict
    dict_output = {}

    # Sample
    for i in range(20, 100, 20):
        dict_output[f"{i}%"] = pd.concat([df_reference, df_target.sample(frac=i/100, replace=False, random_state=seed)])

    dict_output['100%'] = pd.concat([df_reference, df_target])

    return dict_output

#%%
def GenRS_loop(
    reference_name,
    proportion,
    dict_df
    ):

    dict_sampled = {}
    for k, df in dict_df.items():
        dict_sampled[k] = GenRS_sample(dict_df[reference_name], df[df['dataset']=='train'])

    l_reference_df = [v[proportion] for k, v in dict_sampled.items() if k != reference_name]
    l_target_name = [k for k, v in dict_df.items() if k != reference_name]
    l_target_df = [v for k, v in dict_df.items() if k != reference_name]

    for ref_df, target_name, target_df in zip(l_reference_df, l_target_name, l_target_df):
        print('==============================')
        print(f'*** {target_name.upper()} ***')
        print('==============================')

        # While loop to restart training if loss stagnates
        expected_f1 = 0.2
        model_f1 = 0.0
        while model_f1 < expected_f1:
            # Create model
            model = GenRSModel(
                    name = f"{reference_name}_{target_name}_{proportion}",
                    df_reference = ref_df,
                    df_target = target_df
                )
            model.createModel()

            # Extract f1
            df_eval = (model.eval_reference).reset_index()
            model_f1 = df_eval['F1'].iloc[-1]

            try:
                with pd.ExcelWriter(f'results//{reference_name}/GenRS.xlsx', mode='a', if_sheet_exists= 'replace') as writer:
                    (model.eval_reference).to_excel(writer, sheet_name=f'{proportion}-{target_name}')
            except:
                with pd.ExcelWriter(f'results//{reference_name}/GenRS.xlsx') as writer:
                    (model.eval_reference).to_excel(writer, sheet_name=f'{proportion}-{target_name}')

            del model