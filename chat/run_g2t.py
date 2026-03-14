import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Import config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chat.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Convert semantic graphs into descriptions using a seq2seq model.")
    parser.add_argument(
        "--input",
        type=str,
        default=Config.G2T_INPUT,
        help="Input file containing semantic graphs, one per line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=Config.G2T_OUTPUT,
        help="Output file to append generated descriptions to.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=Config.G2T_MODEL,
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=Config.G2T_DEVICE,
        help="Device to run the model on (e.g., cpu, cuda).",
    )
    return parser.parse_args()


args = parse_args()

text = []
with open(args.input, "r") as f:
    for l in f.readlines():
        text.append(l.strip())

model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.model)

with open(args.output, "a") as output_file:
    for i in tqdm(range(len(text))):
        prompt = "Transform the semantic graph into a description. Semantic graph: " + text[i]
        inputs = tokenizer.encode(prompt, max_length=1024, truncation=False, return_tensors="pt").to(args.device)
        outputs = model.generate(inputs, max_length=1024)
        pred = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        output_file.write(pred.strip() + "\n")

