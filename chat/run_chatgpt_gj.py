# 在文件顶部添加导入  
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chat.model_interface import api_model, ModelFactory  
from chat.config import ModelConfig 
from chat.config import dataset_dir, DEFAULT_MAX_CONCURRENT

import aiohttp
import asyncio
import json
import csv
from tqdm.asyncio import tqdm
from datasets import load_dataset

# Set API key and base URL
# api_key = ""
# api_base = ""


def parse_args():
    parser = argparse.ArgumentParser(description="Run graph judgement instruction generation.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. rebel_sub). Can also be set via DATASET_NAME env var.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Datasets root directory. Can also be set via DATASET_ROOT env var.",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default=None,
        help="Dataset folder prefix. Can also be set via DATASET_PREFIX env var.",
    )
    parser.add_argument(
        "--instructions-file",
        type=str,
        default="test_instructions_context_llama2_7b.json",
        help="Instruction JSON filename under the dataset folder.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="pred_instructions_context_llama2_7b_gpt_mini.csv",
        help="Output CSV filename under the dataset folder.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum number of concurrent API calls. Can also be set via MAX_CONCURRENT env var.",
    )
    return parser.parse_args()


args = parse_args()
max_concurrent = args.max_concurrent if args.max_concurrent is not None else DEFAULT_MAX_CONCURRENT

folder = dataset_dir(
    dataset=args.dataset,
    root=args.dataset_root,
    prefix=args.dataset_prefix,
)
# Input and output file paths
input_file = os.path.join(folder, args.instructions_file)
output_file = os.path.join(folder, args.output_file)

# Load instructions from JSON file
total_input = load_dataset("json", data_files=input_file)
data_eval = total_input["train"].train_test_split(
    test_size=499, shuffle=True, seed=42
)["test"]

with open(input_file, "r", encoding="utf-8") as f:
    instructions = json.load(f)

# headers = {
#     "Authorization": f"Bearer {api_key}",
#     "Content-Type": "application/json"
# }

# async def get_chatgpt_completion(session, instruction, input_text):
#     """
#     Send prompt to GPT and get the generated response.
#     """
#     url = f"{api_base}/chat/completions"
#     prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
#     payload = {
#         "model": "gpt-4o-mini",
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0.7
#     }
#     while True:
#         try:
#             async with session.post(url, headers=headers, json=payload) as response:
#                 result = await response.json()
#                 # Return the first completion result
#                 return result["choices"][0]["message"]["content"]
#         except Exception as e:
#             print(e)
#             await asyncio.sleep(1)

async def get_chatgpt_completion(instruction, input_text):  
    """使用统一接口获取GPT完成结果"""  
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"  
      
    return await api_model(  
        prompt=prompt,  
        model_type="graph_judgement",  
        use_http=True,  # 保持原有的HTTP调用方式  
        provider="openai_compatible"  
    )

async def process_instructions():
    """
    Process each instruction and generate responses using GPT with concurrency control.
    """
    semaphore = asyncio.Semaphore(max_concurrent)  # Limit to max concurrent requests
    
    async def limited_get_chatgpt_completion(instruction, input_text):
        async with semaphore:
            return await get_chatgpt_completion(instruction, input_text)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for item in data_eval:
            instruction = item["instruction"]
            input_text = item["input"]
            tasks.append(limited_get_chatgpt_completion(instruction, input_text))

        # Execute all tasks and gather responses
        responses = await tqdm.gather(*tasks)

        # Write responses to a CSV file
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "generated"])  # Write header

            for item, response in zip(instructions, responses):
                prompt = item["instruction"]
                writer.writerow([prompt, response.strip()])

# Run the async process
asyncio.run(process_instructions())
