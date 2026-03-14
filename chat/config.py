# chat/config.py  
import os  
import torch
from typing import Dict, Any  
from pathlib import Path
from typing import Optional


def _env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


class Config:
    """统一配置类，包含所有可配置参数"""

    # 数据集和路径配置
    DATASET_ROOT = _env_str("DATASET_ROOT", "./datasets")
    DATASET_PREFIX = _env_str("DATASET_PREFIX", "GPT4o_mini_result_")
    DATASET_NAME = _env_str("DATASET_NAME", "GenWiki-Hard")
    ITERATION = _env_int("ITERATION", 1)
    MAX_CONCURRENT = _env_int("MAX_CONCURRENT", 8)

    # 模型路径配置（用于 graph_judger）
    BASE_MODEL_PATH = _env_str("BASE_MODEL_PATH", "/data/haoyuhuang/model/llama-3-8b-Instruct/")
    SCIERC_BASE_MODEL_PATH = _env_str("SCIERC_BASE_MODEL_PATH", BASE_MODEL_PATH)
    GENWIKI_BASE_MODEL_PATH = _env_str("GENWIKI_BASE_MODEL_PATH", BASE_MODEL_PATH)
    REBEL_BASE_MODEL_PATH = _env_str("REBEL_BASE_MODEL_PATH", BASE_MODEL_PATH)
    BERT_BASE_MODEL_PATH = _env_str("BERT_BASE_MODEL_PATH", "google-bert/bert-base-uncased")

    # 数据路径配置（用于 graph_judger）
    SCIERC_DATA_PATH = _env_str("SCIERC_DATA_PATH", "data/scierc_4omini_context/train_instructions_context_llama2_7b.json")
    SCIERC_OUTPUT_DIR = _env_str("SCIERC_OUTPUT_DIR", "models/llama3-8b-instruct-lora-scierc-context")

    GENWIKI_DATA_PATH = _env_str("GENWIKI_DATA_PATH", "data/genwiki_4omini_context/train_instructions_context_llama2_7b.json")
    GENWIKI_OUTPUT_DIR = _env_str("GENWIKI_OUTPUT_DIR", "models/llama3-8b-instruct-lora-genwiki-context")

    REBEL_DATA_PATH = _env_str("REBEL_DATA_PATH", "data/rebel_sub_4omini_context/train_instructions_context_llama2_7b.json")
    REBEL_OUTPUT_DIR = _env_str("REBEL_OUTPUT_DIR", "models/llama3-8b-instruct-lora-rebel-context")

    BERT_DATA_PATH = _env_str("BERT_DATA_PATH", "data/scierc_4omini_context/train_instructions_context_bert.json")
    BERT_OUTPUT_DIR = _env_str("BERT_OUTPUT_DIR", "models/bert-classifier-scierc")

    # G2T 配置
    G2T_INPUT = _env_str("G2T_INPUT", "data/GenWiki_source_graph/Iteration1/train.source")
    G2T_OUTPUT = _env_str("G2T_OUTPUT", "outputs/zero_shot_genwiki_results_all/flan_t5_xl_generated_train_I1.target")
    G2T_MODEL = _env_str("G2T_MODEL", "google/flan-t5-xl")
    G2T_DEVICE = _env_str("G2T_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def dataset_root(cls) -> Path:
        return Path(cls.DATASET_ROOT)

    @classmethod
    def dataset_name(cls) -> str:
        return cls.DATASET_NAME

    @classmethod
    def dataset_prefix(cls) -> str:
        return cls.DATASET_PREFIX

    @classmethod
    def dataset_dir(cls, dataset: Optional[str] = None, prefix: Optional[str] = None, root: Optional[str] = None) -> Path:
        if dataset is None:
            dataset = cls.dataset_name()
        if prefix is None:
            prefix = cls.dataset_prefix()
        if root is None:
            root = cls.DATASET_ROOT
        return Path(root) / f"{prefix}{dataset}"

    @classmethod
    def iteration_dir(cls, iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
        return cls.dataset_dir(dataset, **kwargs) / f"Iteration{iteration}"

    @classmethod
    def ensure_iteration_dir(cls, iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
        path = cls.iteration_dir(iteration, dataset, **kwargs)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def test_source_path(cls, dataset: Optional[str] = None, **kwargs) -> Path:
        return cls.dataset_dir(dataset, **kwargs) / "test.source"

    @classmethod
    def test_target_path(cls, dataset: Optional[str] = None, **kwargs) -> Path:
        return cls.dataset_dir(dataset, **kwargs) / "test.target"

    @classmethod
    def denoised_target_path(cls, iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
        return cls.iteration_dir(iteration, dataset, **kwargs) / "test_denoised.target"

    @classmethod
    def entity_path(cls, iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
        return cls.iteration_dir(iteration, dataset, **kwargs) / "test_entity.txt"

    @classmethod
    def graph_output_dir(cls, iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
        return cls.dataset_dir(dataset, **kwargs) / f"Graph_Iteration{iteration}"

    @classmethod
    def graph_output_path(cls, iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
        return cls.graph_output_dir(iteration, dataset, **kwargs) / "test_generated_graphs.txt"

    @classmethod
    def baseline_output_dir(cls, dataset: Optional[str] = None, **kwargs) -> Path:
        return cls.dataset_dir(dataset, **kwargs) / "gpt_baseline"

    @classmethod
    def baseline_output_path(cls, dataset: Optional[str] = None, **kwargs) -> Path:
        return cls.baseline_output_dir(dataset, **kwargs) / "test_generated_graphs.txt"


class ModelConfig:  
    """模型配置类，统一管理所有模型配置"""  
      
    # OpenAI兼容配置  
    OPENAI_COMPATIBLE = {  
        "api_key": os.getenv("API_KEY", ""),  
        "api_base": os.getenv("API_BASE", "https://api-inference.modelscope.cn/v1"),  
        "models": {  
            "entity_extraction": "ZhipuAI/GLM-5",  
            "triple_generation": "deepseek-ai/DeepSeek-V3.2",   
            "baseline": "deepseek-ai/DeepSeek-V3.2",  
            "graph_judgement": "deepseek-ai/DeepSeek-V3.2"  
        },  
        "temperature": {  
            "entity_extraction": 0,  
            "triple_generation": 0.1,  
            "baseline": 0,  
            "graph_judgement": 0.7  
        },  
        "max_concurrent": 8  
    }  
      
    # 其他模型配置可以在这里添加  
    # CLAUDE_CONFIG = {...}  
    # GEMINI_CONFIG = {...}  
      
    @classmethod  
    def get_config(cls, provider: str = "openai_compatible") -> Dict[str, Any]:  
        """获取指定提供商的配置"""  
        configs = {  
            "openai_compatible": cls.OPENAI_COMPATIBLE,  
            # 可以在这里添加其他提供商  
        }  
        return configs.get(provider, cls.OPENAI_COMPATIBLE)



def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


# 默认配置（可用环境变量覆盖）
DEFAULT_DATASET_ROOT = _env_str("DATASET_ROOT", "./datasets")
DEFAULT_DATASET_PREFIX = _env_str("DATASET_PREFIX", "GPT4o_mini_result_")
DEFAULT_DATASET_NAME = _env_str("DATASET_NAME", "GenWiki-Hard")
DEFAULT_ITERATION = _env_int("ITERATION", 1)
DEFAULT_MAX_CONCURRENT = _env_int("MAX_CONCURRENT", 8)


def dataset_root() -> Path:
    return Path(DEFAULT_DATASET_ROOT)


def dataset_name() -> str:
    return DEFAULT_DATASET_NAME


def dataset_prefix() -> str:
    return DEFAULT_DATASET_PREFIX


def dataset_dir(dataset: Optional[str] = None, prefix: Optional[str] = None, root: Optional[str] = None) -> Path:
    if dataset is None:
        dataset = dataset_name()
    if prefix is None:
        prefix = dataset_prefix()
    if root is None:
        root = DEFAULT_DATASET_ROOT
    return Path(root) / f"{prefix}{dataset}"


def iteration_dir(iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
    return dataset_dir(dataset, **kwargs) / f"Iteration{iteration}"


def ensure_iteration_dir(iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
    path = iteration_dir(iteration, dataset, **kwargs)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_source_path(dataset: Optional[str] = None, **kwargs) -> Path:
    return dataset_dir(dataset, **kwargs) / "test.source"


def test_target_path(dataset: Optional[str] = None, **kwargs) -> Path:
    return dataset_dir(dataset, **kwargs) / "test.target"


def denoised_target_path(iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
    return iteration_dir(iteration, dataset, **kwargs) / "test_denoised.target"


def entity_path(iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
    return iteration_dir(iteration, dataset, **kwargs) / "test_entity.txt"


def graph_output_dir(iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
    return dataset_dir(dataset, **kwargs) / f"Graph_Iteration{iteration}"


def graph_output_path(iteration: int, dataset: Optional[str] = None, **kwargs) -> Path:
    return graph_output_dir(iteration, dataset, **kwargs) / "test_generated_graphs.txt"


def baseline_output_dir(dataset: Optional[str] = None, **kwargs) -> Path:
    return dataset_dir(dataset, **kwargs) / "gpt_baseline"


def baseline_output_path(dataset: Optional[str] = None, **kwargs) -> Path:
    return baseline_output_dir(dataset, **kwargs) / "test_generated_graphs.txt"

