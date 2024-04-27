#%%
import openai
import time
import re
import pandas as pd

from key import openai_key
from prompt import *

#%%
def openAIGPT(model, system, user_assistant):
    assert isinstance(system, str), "`system` should be a string"
    assert isinstance(user_assistant, str), "`user_assistant` should be a string"

    system_msg = [{"role": "system", "content": system}]

    user_assistant_msgs = [
        {   "role": "user",
            "content": user_assistant
        }
    ]

    msgs = system_msg + user_assistant_msgs

    try:
        response = openai.ChatCompletion.create(
                    model=model,
                    messages=msgs
                    )
    except:
        time.sleep(30)
        response = openai.ChatCompletion.create(
                    model=model,
                    messages=msgs
                    )

    status_code = response["choices"][0]["finish_reason"]

    assert status_code == "stop", f"The status code was {status_code}."

    return  {'input': user_assistant_msgs,
            'output': response["choices"][0]["message"]["content"]
            }

#%%
def loop_openAIGPT(
    model: str,
    system_prompt: str,
    contents: str
    ):

    results = []
    iteration = 0

    for i in contents:
        print(f"Batch {iteration}")
        batch_results = openAIGPT(model, system_prompt, i)
        results.append(batch_results)
        iteration += 1
        if iteration != 0 and iteration % 20 == 0:
            print("Pausing for 30 sec")
            time.sleep(30)
    
    return results

#%%
def cleanOutput(
    results: list,
    emotions: list
    ):

    ref = {i:emo for i,emo in enumerate(emotions)}

    input_clean = []
    results_clean = []
    for result in results:
        # Clean Output
        clean = result['output']\
                    .replace(', "', ", '")\
                    .replace('" ', "', ")\
                    .replace("'", '')\
                    .replace('100%', '1.0')\
                    .replace('50%', '0.5')\
                    .replace('\n\n', '\n')
        clean = re.sub(r'(?<=[,])(?=[^\s])', r' ', clean)
        results_clean.append(clean)

        # Extract Output
        input_clean.append(result['input'][0]['content'].splitlines())

    # Extact Data
    pattern = re.compile(r"\[.*?\]")
    idx = 0
    issues = []
    clean_output = []
    for r in results_clean:
        # Apply to each line
        print(idx)
        for line in r.splitlines():
            # Identify pattern and extract output
            matches = pattern.findall(line)
            # Reshape
            matches_clean = []
            for m_list in matches:
                matches_clean.append(m_list.replace("'", '').strip('[').strip(']').split(', '))
            # Map to emotions
            matches_dic = {str(k):0 for k in list(ref.keys())}
            for j in matches_clean:
                try:
                    # Check if there are errors in response
                    if ref[int(j[0])] != j[1].replace(' ', ''):
                        # No emotion, only label and prob
                        if len(j) < 3:
                            issues.append([idx, line, 'Incorrect format'])
                        # Emotion not in prescribed list
                        elif j[1] not in list(ref.values()):
                            issues.append([idx, line, 'Incorrect emotion'])
                        # Incorrect label for emotion
                        else:
                            # Find correct label for emotion and replace
                            x = str([x for x in ref if ref[x]==j[1].replace(' ', '')][0])
                            issues.append([idx, line, 'Incorrect label', x])
                            j[0] = x
                    # Add to dictionary
                    matches_dic[j[0]] = float(j[2])
                except:
                    pass
                output = list(matches_dic.values())
            clean_output.append(output)
        idx += 1

    df_output = pd.DataFrame(clean_output)

    return df_output

#%%
class openAIGPT():
    def __init__(self, data, dataset_name, text_col='text', key=openai_key):
        self.df, self.dataset_name, self.text_col = data.copy(), dataset_name, text_col
        self.system_prompt, self.emotions = create_prompt(self.df, self.dataset_name)
        self.contents = parse_content(self.df)
        openai.api_key = key

    def GPT3_5(self):
        self.GPT3_5 = loop_openAIGPT("gpt-3.5-turbo", self.system_prompt, self.contents)

    def GPT3_5_clean(self):
        self.GPT3_5 = cleanOutput(self.GPT3_5, self.emotions)

    def GPT4(self):
        self.GPT4 = loop_openAIGPT("gpt-4", self.system_prompt, self.contents)

    def GPT3_5_clean(self):
        self.GPT4 = cleanOutput(self.GPT4, self.emotions)