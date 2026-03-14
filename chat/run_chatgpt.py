# 在文件顶部添加导入  
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chat.model_interface import api_model  
from chat.config import ModelConfig  
from chat.config import (
    test_target_path,
    baseline_output_dir,
    baseline_output_path,
    DEFAULT_MAX_CONCURRENT,
)

import argparse
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# api_key = ""
# api_base = ""

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a semantic graph for a dataset.")
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
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum number of concurrent API calls. Can also be set via MAX_CONCURRENT env var.",
    )
    return parser.parse_args()


args = parse_args()
max_concurrent = args.max_concurrent if args.max_concurrent is not None else DEFAULT_MAX_CONCURRENT

# Read the text
input_file = test_target_path(
    dataset=args.dataset,
    root=args.dataset_root,
    prefix=args.dataset_prefix,
)
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Expected input file at {input_file}, but it does not exist.")

with open(input_file, "r") as f:
    text = [l.strip() for l in f.readlines()]

# Ensure output directory exists
os.makedirs(
    baseline_output_dir(dataset=args.dataset, root=args.dataset_root, prefix=args.dataset_prefix),
    exist_ok=True,
)

# 修改API调用

# async def api_model(
#     prompt, system_prompt=None, history_messages=[], **kwargs
# ) -> str:
#     openai_async_client = AsyncOpenAI(
#         api_key=api_key,
#         base_url=api_base
#     )
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.extend(history_messages)
#     messages.append({"role": "user", "content": prompt})
    
#     response = await openai_async_client.chat.completions.create(
#         model="gpt-4o", messages=messages, temperature=0, **kwargs
#     )

#     return response.choices[0].message.content

# async def api_model(
#     prompt, system_prompt=None, history_messages=[], **kwargs
# ) -> str:

#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.extend(history_messages)
#     messages.append({"role": "user", "content": prompt})
    
#     response = await api_model(  
#         prompt=prompt,  
#         system_prompt=system_prompt,  
#         history_messages=history_messages,  
#         model_type="baseline",  
#         **kwargs  
#     )

#     return response.choices[0].message.content

async def _run_api(queries, max_concurrent=DEFAULT_MAX_CONCURRENT):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_api_model(query):
        async with semaphore:
            return await api_model(query, model_type="baseline")

    tasks = [limited_api_model(query) for query in queries]
    answers = await tqdm.gather(*tasks)
    return answers

async def process_texts():
    queries = []
    for t in text:
        prompt = (
            f"""
Transform the text into a semantic graph.

Example#1:
Text: Shotgate \"Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.\"
Semantic Graph: 
```[["Shotgate Thickets", "instance of", "Nature reserve"], ["Shotgate Thickets", "country", "United Kingdom"], ["Shotgate Thickets", "operator", "Essex Wildlife Trust"]]```
Example#2:
Text: The Eiffel Tower, located in Paris, France, is a famous landmark and a popular tourist attraction. It was designed by the engineer Gustave Eiffel and completed in 1889.
Semantic Graph:
```[["Eiffel Tower", "located in", "Paris"], ["Eiffel Tower", "located in", "France"], ["Eiffel Tower", "instance of", "landmark"], ["Eiffel Tower", "attraction type", "tourist attraction"],["Eiffel Tower", "designed by", "Gustave Eiffel"], ["Eiffel Tower", "completion year", "1889"]]```

Note: 
1. Make sure each item in the list is a triple with strictly three items.
2. Give me the answer in the form of a semantic graph extactly as example shown, which is a list of lists.
3. DO NOT output like '```json[ .. ]```', SHOUDLE BE LIKE '```[...]```'.

Text: \"{t}\"
Semantic Graph:
"""
        )
        queries.append(prompt)

    responses = await _run_api(queries, max_concurrent=max_concurrent)

    output_file = baseline_output_path(
        dataset=args.dataset, root=args.dataset_root, prefix=args.dataset_prefix
    )
    with open(output_file, "w") as output_file_obj:
        for response in responses:
            output_file_obj.write(response.strip().replace('\n', '') + '\n')

# Run the async process
asyncio.run(process_texts())
