# 在文件顶部添加导入  
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chat.model_interface import api_model  
from chat.config import ModelConfig 
from chat.config import (
    denoised_target_path,
    entity_path,
    graph_output_dir,
    graph_output_path,
    DEFAULT_MAX_CONCURRENT,
)

import asyncio
import json
import functools
import ast
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# Set API key and base URL

# api_key = ""
# api_base = ""
# openai_async_client = AsyncOpenAI(api_key=api_key, base_url=api_base)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a semantic graph (triples) from denoised text and entities.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. GenWiki-Hard). Can also be set via DATASET_NAME env var.",
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
        "--denoised-iteration",
        type=int,
        default=1,
        help="Which iteration's denoised files to read (default 1).",
    )
    parser.add_argument(
        "--graph-iteration",
        type=int,
        default=1,
        help="Which graph iteration directory to write output into (default 1).",
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

# Read the text to be denoised
text_file = denoised_target_path(
    args.denoised_iteration,
    dataset=args.dataset,
    root=args.dataset_root,
    prefix=args.dataset_prefix,
)
if not os.path.exists(text_file):
    raise FileNotFoundError(f"Expected denoised text file at {text_file}, but it does not exist.")

with open(text_file, "r") as f:
    text = [l.strip() for l in f.readlines()]

# Read the corresponding entities
entity_file = entity_path(
    args.denoised_iteration,
    dataset=args.dataset,
    root=args.dataset_root,
    prefix=args.dataset_prefix,
)
if not os.path.exists(entity_file):
    raise FileNotFoundError(f"Expected entity file at {entity_file}, but it does not exist.")

with open(entity_file, "r") as f:
    entity = [l.strip() for l in f.readlines()]

# Ensure output directory exists
os.makedirs(
    graph_output_dir(
        args.graph_iteration,
        dataset=args.dataset,
        root=args.dataset_root,
        prefix=args.dataset_prefix,
    ),
    exist_ok=True,
)

# 修改API调用

# async def api_model(prompt, **kwargs):
#     messages = [{"role": "user", "content": prompt}]
#     response = await openai_async_client.chat.completions.create(
#         model="gpt-3.5-turbo", messages=messages, temperature=0.1, **kwargs
#     )
#     return response.choices[0].message.content

# async def api_model(prompt, **kwargs):
#     messages = [{"role": "user", "content": prompt}]
#     response = await api_model(  
#     prompt=prompt,  
#     model_type="triple_generation",  
#     **kwargs  
#     )
#     return response.choices[0].message.content


async def _run_api(prompts, max_concurrent=DEFAULT_MAX_CONCURRENT):
    semaphore = asyncio.Semaphore(max_concurrent)
    async def limited_api_model(prompt):
        async with semaphore:
            return await api_model(prompt, model_type="triple_generation")
    tasks = [limited_api_model(prompt) for prompt in prompts]
    answers = await tqdm.gather(*tasks)
    return answers

async def main():
    prompts = []
    for i in range(len(text)):
        prompt = (
                f"Goal:\nTransform the text into a semantic graph(a list of triples) with the given text and entities. "
                f"In other words, You need to find relations between the given entities with the given text.\n"
                f"Attention:\n1.Generate triples as many as possible. "
                f"2.Make sure each item in the list is a triple with strictly three items.\n\n"
                f"Here are two examples:\n"
                f"Example#1: \nText: \"Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.\"\n"
                f"Entity List: [\"Shotgate Thickets\", \"Nature reserve\", \"United Kingdom\", \"Essex Wildlife Trust\"]\n"
                f"Semantic Graph: [[\"Shotgate Thickets\", \"instance of\", \"Nature reserve\"], "
                f"[\"Shotgate Thickets\", \"country\", \"United Kingdom\"], [\"Shotgate Thickets\", \"operator\", \"Essex Wildlife Trust\"]]\n"
                f"Example#2:\nText: \"The Eiffel Tower, located in Paris, France, is a famous landmark and a popular tourist attraction. "
                f"It was designed by the engineer Gustave Eiffel and completed in 1889.\"\n"
                f"Entity List: [\"Eiffel Tower\", \"Paris\", \"France\", \"landmark\", \"Gustave Eiffel\", \"1889\"]\n"
                f"Semantic Graph: [[\"Eiffel Tower\", \"located in\", \"Paris\"], [\"Eiffel Tower\", \"located in\", \"France\"], "
                f"[\"Eiffel Tower\", \"instance of\", \"landmark\"], [\"Eiffel Tower\", \"attraction type\", \"tourist attraction\"], "
                f"[\"Eiffel Tower\", \"designed by\", \"Gustave Eiffel\"], [\"Eiffel Tower\", \"completion year\", \"1889\"]]\n\n"
                f"Refer to the examples and here is the question:\nText: {text[i]}\nEntity List:{entity[i]}\nSemantic graph:"
            )
        prompts.append(prompt)

    responses = await _run_api(prompts, max_concurrent=max_concurrent)

    # 写入文件
    output_file = graph_output_path(
        args.graph_iteration,
        dataset=args.dataset,
        root=args.dataset_root,
        prefix=args.dataset_prefix,
    )
    with open(output_file, "w") as output_file_obj:
        for response in responses:
            output_file_obj.write(response.strip().replace('\n', '') + '\n')

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
