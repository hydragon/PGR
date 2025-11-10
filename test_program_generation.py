import argparse
import json
import os
import time
import re
import pickle

import requests

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
def extract_code_from_string(input_string):
    match = re.search(r'```python\n(.*?)```', input_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None
    
def chat_responce(model_name='deepseek-chat', messages=[], system_prompt=[{'role': 'system', 'content': 'You are the powerful GPT-4 created by OpenAI.'}], max_tokens=2048, temperature=0.7, top_p=0.1):
    url = 'https://api.deepseek.com/chat/completions'
    headers = {"Content-Type": "application/json", "Authorization": 'Bearer your-key'}

    messages = system_prompt + messages

    data = {"model":model_name, "messages":messages, "stream": False, "max_tokens": max_tokens, "temperature": temperature, "top_p":top_p}
    resp_json= requests.post(url, headers=headers, json=data, timeout=120).json()

    responce = resp_json['choices'][0]['message']['content']

    return responce
    
def get_answer(model_name: str, qid: int, claim: str, gt_entities: list, KG: dict, max_tokens: int):
    prompt_w_entity = open_file('./prompts/prompt_w_entity.txt').replace('<<<<S>>>>', claim).replace('<<<<ENTITIES>>>>', str(gt_entities))
    
    # program generation
    for _ in range(3):
        try:
            program_gen_result = chat_responce(model_name=model_name, messages=[{
                        "role": "user",
                        "content": prompt_w_entity,
                    }], system_prompt=[{"role": "system", "content": "You are a helpful assistant."}], max_tokens=max_tokens, temperature=0.2, top_p=0.1)
            program_gen_result = program_gen_result
            program_gen_dict = {'claim':claim, 'entities':gt_entities, 'program':program_gen_result}
            break
        except Exception as e:
            print("[ERROR]", e)
            time.sleep(5)
    with open(f'./result/program_generate/{qid}_program.json', 'w') as f:
        json.dump(program_gen_dict, f, indent=4)

    return 'True'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing input arguments.")
    parser.add_argument('--model', type=str, required=True, help='Model name. e.g. gpt-4, gpt-3.5-turbo, deepseek-chat')
    parser.add_argument('--test', type=str, required=True, help='Test dataset path.')
    
    args = parser.parse_args()

    model_name = args.model
    test_path = args.test

    dbp = None
    
    start_token = 0
    
    ####For new experiment, use it.
    questions_dict = {}
    entity_set_dict = {}
    label_set_dict = {}
    with open(test_path) as f:
        for line in f:
            if not line:
                continue
            q = json.loads(line)
            questions_dict[q["question_id"]] = q["question"]
            entity_set_dict[q["question_id"]] = q["entity_set"]
            label_set_dict[q["question_id"]] = q["Label"]

    Error = []

    for qid, question in questions_dict.items():
        try:                
            return_result = get_answer(model_name, qid, question, entity_set_dict[qid], KG=dbp, max_tokens=1024)
            print(qid, ': Correct!')
        except:
            Error.append(qid)
            print(qid, ': Error...')
