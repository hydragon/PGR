import argparse
import json
import os
import time
import re
import pickle

import requests

from kg_program_bi_reverse import execute_program

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
def extract_code_from_string(input_string):
    match = re.search(r'```python\n(.*?)```', input_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return input_string
    
def get_answer(model_name: str, qid: int, claim: str, gt_entities: list, KG: dict, max_tokens: int):
    
    if os.path.exists(f'./result/program_generate/{qid}_program.json'):
        pass
    else:
        return "No Program"
    with open(f'./result/program_generate/{qid}_program.json', 'r') as f:
        program_gen_dict = json.load(f)

    program_extracted = extract_code_from_string(program_gen_dict['program'])
    predicted = execute_program(program_extracted, KG, claim, qid)

    return predicted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing input arguments.")
    parser.add_argument('--model', type=str, required=True, help='Model name. e.g. gpt-4, deepseek-chat')
    parser.add_argument('--kg', type=str, required=True, help='Path of KG.')
    parser.add_argument('--test', type=str, required=True, help='Test dataset path.')
    
    args = parser.parse_args()

    model_name = args.model
    kg_path = args.kg
    test_path = args.test

    print('Start KG loading...')
    with open(kg_path, 'rb') as f:
        dbp = pickle.load(f)

    print('-----KG loaded-----')
    
    final_results = []
    start_token = 0
    
    ####For new experiment, use it.
    result = {}
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

    Correct = []
    Wrong = []
    Error = []
    Another = []

    for qid, question in questions_dict.items():
        try:
            final_r = get_answer(model_name, qid, question, entity_set_dict[qid], KG=dbp, max_tokens=1024)
            final_results.append(final_r)
            if(final_r == "No Program"):
                continue
            if final_r == 'Another Answer':
                Another.append(qid)
                print(qid, ': ', final_r)
                result[qid] = final_r
            elif final_r == label_set_dict[qid][0]:
                Correct.append(qid)
                print(qid, ': Correct!')
                result[qid] = 'Correct'
            else:
                Wrong.append(qid)
                print(qid, ': Wrong...')
                result[qid] = 'Wrong'
        except:
            Error.append(qid)
            print(qid, ': Error...')
            result[qid] = 'Error'
        with open(f'./result_final.pickle', 'wb') as f:
            pickle.dump(result, f)

    
    tot_corr = 0
    for tot_id in list(result):
        if result[tot_id] == 'Correct':
            tot_corr += 1
    
    print('Accuracy: ', tot_corr/len(list(result)))