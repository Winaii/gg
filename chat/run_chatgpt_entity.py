# 在文件顶部添加导入  
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chat.model_interface import api_model  
from chat.config import ModelConfig, Config
from chat.config import (
    dataset_dir,
    test_target_path,
    denoised_target_path,
    ensure_iteration_dir,
    entity_path,
    DEFAULT_MAX_CONCURRENT,
)

import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract entities and denoise text for a dataset iteration."
    )
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
        "--iteration",
        type=int,
        default=None,
        help="Iteration index. Iteration 1 reads base test.target; iteration >1 reads previous iteration's denoised target.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum number of concurrent API calls. Can also be set via MAX_CONCURRENT env var.",
    )
    return parser.parse_args()


from typing import Optional


def load_text(dataset: str, iteration: int, dataset_root: Optional[str], dataset_prefix: Optional[str]):
    # Iteration 1 reads base test.target; later iterations read previous iteration's denoised output.
    if iteration == 1:
        input_path = test_target_path(dataset=dataset, root=dataset_root, prefix=dataset_prefix)
    else:
        input_path = denoised_target_path(
            iteration - 1, dataset=dataset, root=dataset_root, prefix=dataset_prefix
        )

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Expected input file at {input_path}, but it does not exist."
        )

    with open(input_path, "r") as f:
        return [l.strip() for l in f.readlines()]


args = parse_args()
Iteration = args.iteration if args.iteration is not None else 1
max_concurrent = (
    args.max_concurrent if args.max_concurrent is not None else DEFAULT_MAX_CONCURRENT
)

# Load the input text for this iteration
text = load_text(
    dataset=args.dataset,
    iteration=Iteration,
    dataset_root=args.dataset_root,
    dataset_prefix=args.dataset_prefix,
)

# 修改API调用

async def _run_api(queries, max_concurrent=8):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_api_model(query):
        async with semaphore:
            # await asyncio.sleep(15)  # 添加延迟以避免过快的请求
            return await api_model(query, model_type="entity_extraction")

    tasks = [limited_api_model(query) for query in queries]
    answers = await tqdm.gather(*tasks)
    return answers

async def extract_entities(texts):
    prompts = []
    for t in texts:
        prompt = f"""
Goal:
Transform the text into a list of entities.

Here are two examples:
Example#1:
Text: "Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust."
List of entities: ["Shotgate Thickets", "Nature reserve", "United Kingdom", "Essex Wildlife Trust"]
Example#2:
Text: "Garczynski Nunatak () is a cone-shaped nunatak, the highest in a cluster of nunataks close west of Mount Brecher, lying at the north flank of Quonset Glacier in the Wisconsin Range of the Horlick Mountains of Antarctica. It was mapped by the United States Geological Survey from surveys and U.S. Navy air photos, 1959–60, and was named by the Advisory Committee on Antarctic Names for Carl J. Garczynski, a meteorologist in the Byrd Station winter party, 1961."
List of entities: ["Garczynski Nunatak", "nunatak", "Wisconsin Range", "Mount Brecher", "Quonset Glacier", "Horlick Mountains"]

Refer to the examples and here is the question:
Text: "{t}"
List of entities: """
        prompts.append(prompt)
    
    entities_list = await _run_api(prompts, max_concurrent=max_concurrent)
    return entities_list

async def denoise_text(texts, entities_list):
    prompts = []
    for t, entities in zip(texts, entities_list):
        prompt = f"""
Goal:
Denoise the raw text with the given entities, which means remove the unrelated text and make it more formatted.

Here are two examples:
Example#1:
Raw text: "Zakria Rezai (born 29 July 1989) is an Afghan footballer who plays for Ordu Kabul F.C., which is a football club from Afghanistan. He is also an Afghanistan national football team player, and he has 9 caps in the history. He wears number 14 on his jersey and his position on field is centre back."
Entities: ["Zakria Rezai", "footballer", "Ordu Kabul F.C.", "Afghanistan", "29 July 1989"]
Denoised text: "Zakria Rezai is a footballer. Zakria Rezai is a member of the sports team Ordu Kabul F.C. Zakria Rezai has the citizenship of Afghanistan. Zakria Rezai was born on July 29, 1989. Ordu Kabul F.C. is a football club. Ordu Kabul F.C. is based in Afghanistan."
Example#2:
Raw text: "Elizabeth Smith, a renowned British artist, was born on 12 May 1978 in London. She is specialized in watercolor paintings and has exhibited her works in various galleries across the United Kingdom. Her most famous work, 'The Summer Breeze,' was sold at a prestigious auction for a record price. Smith is also a member of the Royal Society of Arts and has received several awards for her contributions to the art world."
Entities: ["Elizabeth Smith", "British artist", "12 May 1978", "London", "watercolor paintings", "United Kingdom", "The Summer Breeze", "Royal Society of Arts"]
Denoised text: "Elizabeth Smith is a British artist. Elizabeth Smith was born on May 12, 1978. Elizabeth Smith was born in London. Elizabeth Smith specializes in watercolor paintings. Elizabeth Smith's artwork has been exhibited in the United Kingdom. 'The Summer Breeze' is a famous work by Elizabeth Smith. Elizabeth Smith is a member of the Royal Society of Arts."

Refer to the examples and here is the question:
Raw text: {t}
Entities: {entities}
Denoised text: """
        prompts.append(prompt)
    
    denoised_texts = await _run_api(prompts, max_concurrent=max_concurrent)
    return denoised_texts



async def main():
    # 提取实体并保存
    entities_list = await extract_entities(text)

    # Ensure this iteration directory exists.
    ensure_iteration_dir(
        Iteration, dataset=args.dataset, root=args.dataset_root, prefix=args.dataset_prefix
    )

    entity_file = entity_path(
        Iteration, dataset=args.dataset, root=args.dataset_root, prefix=args.dataset_prefix
    )
    with open(entity_file, "w") as output_file:
        for entities in entities_list:
            output_file.write(entities.strip().replace('\n', '') + '\n')

    # 读取提取的实体
    last_extracted_entities = []
    with open(entity_file, "r") as f:
        for l in f.readlines():
            last_extracted_entities.append(l.strip())

    # 去噪文本并保存
    denoised_texts = await denoise_text(text, last_extracted_entities)
    denoised_file = denoised_target_path(
        Iteration, dataset=args.dataset, root=args.dataset_root, prefix=args.dataset_prefix
    )
    with open(denoised_file, "w") as output_file:
        for denoised_text in denoised_texts:
            output_file.write(denoised_text.strip().replace('\n', '') + '\n')

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())
