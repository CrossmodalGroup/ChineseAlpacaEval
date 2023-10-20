import os
import re
import time
import json
import openai
import random
import tiktoken
import argparse
from tqdm import tqdm
import numpy as np


tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
SYSTEM_CONTENT = '''You are a helpful assistant that ranks models according to the quality of their Chinese answers.'''


def set_seed(args):
    random.seed(args.seed)


def extract_last_brackets(text):
    pattern = r'\[([^\[\]]+)\]'
    matches = re.findall(pattern, text)
    if matches:
        return '[' + matches[-1].strip() + ']'
    else:
        return None


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(tiktoken_encoding.encode(string))
    return num_tokens


def query_openai(
                model,
                prompt,
                system_content=SYSTEM_CONTENT,
                max_tokens=2048,
                temperature=0.0,
                top_p=1.0
    ):
    max_tries = 10
    num_tries = 0
    success = False
    response = None

    while num_tries < max_tries and not success:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            success = True
        except Exception as e:
            print(f"Error encountered: {e}")
            print(f'key:{openai.api_key}')
            num_tries += 1
            if num_tries == max_tries:
                print("Maximum number of tries reached. Aborting.")
                raise
            print(f"Retrying (attempt {num_tries}/{max_tries})...")
            time.sleep(10)
    content = response["choices"][0]["message"]["content"]
    time.sleep(16)
    return content.strip()


def evaluate(args, evaluator, model_outputs, reference_outputs):
    if len(model_outputs) != len(reference_outputs):
        raise Exception("The number of responses must be equal to the number of instructions.")

    for i in tqdm(range(len(reference_outputs))):
        response1 = model_outputs[i]
        response2 = reference_outputs[i]
        if response2['instruction'] != response1['instruction']:
            raise Exception("Incorrect response order.")
        instruction = response2['instruction']

        # Randomly swap positions to prevent position bias.
        swap = random.random() < 0.5
        if swap:
            tmp = response1
            response1 = response2
            response2 = tmp

        prompt = PROMPT.format(instruction=instruction, output_1=response1['response'], output_2=response2['response'])
        tokens = num_tokens_from_string(prompt)
        response1_tokens = num_tokens_from_string(model_outputs[i]['response'])
        # Checks if the input length exceeds the maximum length allowed by evaluator.
        if tokens > 8000 and evaluator == 'gpt-4-0613':
            max_tokens_num = 8000 - (tokens - response1_tokens)
            raise Exception(f"The prompt is too long, please reduce the length of your model's response to {max_tokens_num} tokens (now {response1_tokens}).")
        elif tokens > 4000:
            max_tokens_num = 4000 - (tokens - response1_tokens)
            raise Exception(f"The prompt is too long, please reduce the length of your model's response to {max_tokens_num} tokens (now {response1_tokens}).")

        eval_result = query_openai(evaluator, prompt)
        eval_result = extract_last_brackets(eval_result)
        if eval_result:
            try:
                result = eval(eval_result)
                if result[0]['rank'] == 1 or result[0]['rank'] == '1':
                    result = {"result": result[0]['model']}
                elif result[1]['rank'] == 1 or result[1]['rank'] == '1':
                    result = {"result": result[1]['model']}
                else:
                    result = {"result": "error"}
            except:
                result = {"result": "error"}

            # Switch it back.
            if swap and result['result'] != 'error':
                swap_dict = {'model_1':'model_2', 'model_2':'model_1'}
                result['result'] = swap_dict[result['result']]
        else:
            result = {"result": "error"}

        with open(args.result_file, 'a+') as f:
            result['evaluator'] = args.evaluator
            result['reference_model'] = args.reference
            result['instruction'] = instruction
            result['model_output'] = model_outputs[i]['response']
            result['reference_output'] = reference_outputs[i]['response']
            f.write(json.dumps(result, ensure_ascii=False))
            f.write('\n')


def statistic(args):
    with open(args.result_file, 'r') as f:
        win = [0, 0, 0]
        for line in f.readlines():
            data = json.loads(line)
            if data['result'] == 'model_1':
                win[0] += 1
            elif data['result'] == 'model_2':
                win[2] += 1
            else:
                win[1] += 1

        result_dict = {
            'win': win[0], 'lose': win[2], 'win_rate': np.round(win[0]/len(reference_outputs), 4), 
            'lose_rate': np.round(win[2]/len(reference_outputs), 4), 'error': win[1], 
            'error_rate': np.round(win[1]/len(reference_outputs), 4)
        }
        print('Evaluation results:\n')
        print(result_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--model_name", default='gpt-4-0613', type=str)
    parser.add_argument("--reference", default='text-davinci-003', choices=['gpt-4-0613', 'gpt-3.5-turbo-0613', 'text-davinci-003'], type=str)
    parser.add_argument("--evaluator", default='gpt-4-0613', choices=['gpt-4-0613', 'gpt-3.5-turbo-0613'], type=str)

    args = parser.parse_args()
    set_seed(args)

    # Get model_outputs file and result_file
    args.model_outputs = f'./model_outputs/{args.model_name}.jsonl'
    args.result_file = f'./results/{args.model_name}_vs_{args.reference}.jsonl'

    # Get prompt for evaluation
    with open('./data/eval_prompt_zh.txt', 'r') as f:
        PROMPT = f.read()

    # Get outputs of the target model
    model_outputs = []
    with open(args.model_outputs, 'r') as f:
        for line in f.readlines():
            model_outputs.append(json.loads(line))

    # Get outputs of the reference model
    reference_outputs = []
    reference_file = f'./model_outputs/{args.reference}.jsonl'
    with open(reference_file, 'r') as f:
        for line in f.readlines():
            reference_outputs.append(json.loads(line))

    # Evaluate
    evaluate(args, args.evaluator, model_outputs, reference_outputs)

    # Statistic
    statistic(args)
