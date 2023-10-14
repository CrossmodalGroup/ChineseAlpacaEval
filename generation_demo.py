import os
import json
import time
import openai
import random
import tiktoken
import argparse
import numpy as np
from tqdm import tqdm


random.seed(42)
SYSTEM_CONTENT_zh = '''您是一位乐于助人的智能助手。请用中文回复用户的请求。'''


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def query_openai(
                args,
                model,
                prompt,
                system_content=SYSTEM_CONTENT_zh
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
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p
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
    return content


def generate(args, model_name):
    data_list = []
    # Read ChineseAlpacaEval instruction set
    with open('./data/chinese_alpaca_eval.jsonl', 'r') as f:
        for line in f.readlines():
            data_list.append(json.loads(line))

    # Query model to get outputs
    output_file = f'./model_outputs/{model_name}.jsonl'
    for i in tqdm(range(len(data_list))):
        data = data_list[i]
        instruction = data['instruction_zh']
        response = query_openai(args, model_name, instruction)
        print(response)

        # Write to model outputs file
        with open(output_file, 'a+', encoding='utf-8') as f:
            output_dict = {
                'instruction':instruction, 'response':response
            }
            f.write(json.dumps(output_dict, ensure_ascii=False))
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    # decoding
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--length_penalty", default=1, type=int)
    parser.add_argument("--repetition_penalty", default=1, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)

    args = parser.parse_args()
    set_seed(args)

    generate(args, model)
