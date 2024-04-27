#%%
import pandas as pd
import numpy as np

#%%
def create_prompt(
    data: pd.DataFrame,
    dataset_name: str
    ) -> str:

    df = data.copy()

    # Get emotion details
    l_emotions = [emo for emo in df.columns if emo not in ['source', 'dataset', 'id', 'text', 'text_clean']]
    s_emotions = '\n'.join([f"{str(i)}: '{emo}'" for i,emo in enumerate(l_emotions)]) + '\n'

    multiemo = np.max(df[l_emotions].sum(axis=1))
    if multiemo > 1:
        s_multiemo = 'Each example can have ' + \
                        ', '.join([str(i) for i in range(1, multiemo)]) + \
                        f', or {multiemo} emotions.\nNot all of them have {multiemo} emotions.\n'
        l_multiemo = 'List the emotions you identified individually in the following format for each example: [' +\
                        '], ['.join([f'label{i+1}, emotion{i+1}, probability{i+1}' for i,n in enumerate(l_emotions)]) +\
                        ']\n'
    else:
        s_multiemo = 'Each example can only have 1 emotion.\n'
        l_multiemo = 'List the emotions you identified individually in the following format for each example: [label1,  emotion1, probability1]\n'

    # Create initial prompt
    system_prompt = f"Identify emotions based on the {dataset_name} dataset with {len(l_emotions)} emotions\n" +\
        "Below are the labels and corresponding emotions:\n" +\
        s_emotions +\
        '\n' +\
        s_multiemo +\
        '\n' +\
        l_multiemo +\
        '\n' +\
        "I will provide the examples now"

    return system_prompt, l_emotions

#%%
def parse_content(
    data: pd.DataFrame,
    text_col: str = 'text',
    n_items: int = 20
    ) -> list:

    df = data.copy()

    # Create batches
    start = 0
    end = len(df[text_col])
    step = n_items
    batches = [df[text_col][x : x+step] for x in range(start, end, step)]

    # Create content
    numbered = [[f"{n}. " + item for item, n in zip(batch, range(1, len(batch)+1))] for batch in batches]
    contents = [i for i in [f"\n".join(j) for j in numbered]]

    return contents