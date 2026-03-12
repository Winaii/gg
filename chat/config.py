# chat/config.py  
import os  
from typing import Dict, Any  
  
class ModelConfig:  
    """模型配置类，统一管理所有模型配置"""  
      
    # OpenAI兼容配置  
    OPENAI_COMPATIBLE = {  
        "api_key": os.getenv("API_KEY", ""),  
        "api_base": os.getenv("API_BASE", "https://api.openai.com/v1"),  
        "models": {  
            "entity_extraction": "gpt-4o-mini",  
            "triple_generation": "gpt-3.5-turbo",   
            "baseline": "gpt-4o",  
            "graph_judgement": "gpt-4o-mini"  
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
