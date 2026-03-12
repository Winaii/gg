# chat/model_interface.py  
import asyncio  
import abc  
from typing import List, Dict, Any, Optional  
from openai import AsyncOpenAI  
import aiohttp  
from .config import ModelConfig  
  
class BaseModelInterface(abc.ABC):  
    """模型接口抽象基类"""  
      
    def __init__(self, config: Dict[str, Any]):  
        self.config = config  
        self.api_key = config["api_key"]  
        self.api_base = config["api_base"]  
      
    @abc.abstractmethod  
    async def chat_completion(self,   
                            messages: List[Dict[str, str]],   
                            model: str,  
                            temperature: float = 0,  
                            **kwargs) -> str:  
        """聊天完成接口"""  
        pass  
  
class OpenAICompatibleModel(BaseModelInterface):  
    """OpenAI兼容模型实现"""  
      
    def __init__(self, config: Dict[str, Any]):  
        super().__init__(config)  
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)  
      
    async def chat_completion(self,   
                            messages: List[Dict[str, str]],   
                            model: str,  
                            temperature: float = 0,  
                            **kwargs) -> str:  
        """使用AsyncOpenAI客户端进行聊天完成"""  
        response = await self.client.chat.completions.create(  
            model=model,   
            messages=messages,   
            temperature=temperature,   
            **kwargs  
        )  
        return response.choices[0].message.content  
  
class OpenAICompatibleModelHTTP(BaseModelInterface):  
    """OpenAI兼容模型的HTTP实现（用于run_chatgpt_gj.py）"""  
      
    async def chat_completion(self,   
                            messages: List[Dict[str, str]],   
                            model: str,  
                            temperature: float = 0,  
                            **kwargs) -> str:  
        """使用aiohttp进行HTTP请求"""  
        headers = {  
            "Authorization": f"Bearer {self.api_key}",  
            "Content-Type": "application/json"  
        }  
          
        payload = {  
            "model": model,  
            "messages": messages,  
            "temperature": temperature  
        }  
          
        url = f"{self.api_base}/chat/completions"  
          
        while True:  
            try:  
                async with aiohttp.ClientSession() as session:  
                    async with session.post(url, headers=headers, json=payload) as response:  
                        result = await response.json()  
                        return result["choices"][0]["message"]["content"]  
            except Exception as e:  
                print(f"请求失败: {e}")  
                await asyncio.sleep(1)  
  
class ModelFactory:  
    """模型工厂类"""  
      
    @staticmethod  
    def create_model(provider: str = "openai_compatible",   
                    use_http: bool = False) -> BaseModelInterface:  
        """创建模型实例"""  
        config = ModelConfig.get_config(provider)  
          
        if use_http:  
            return OpenAICompatibleModelHTTP(config)  
        else:  
            return OpenAICompatibleModel(config)  
  
# 便捷函数，用于替换原有的api_model函数  
async def api_model(prompt: str,   
                   system_prompt: Optional[str] = None,  
                   history_messages: List[Dict[str, str]] = [],  
                   model_type: str = "entity_extraction",  
                   provider: str = "openai_compatible",  
                   use_http: bool = False,  
                   **kwargs) -> str:  
    """统一的API调用函数"""  
      
    config = ModelConfig.get_config(provider)  
    model = config["models"][model_type]  
    temperature = config["temperature"][model_type]  
      
    model_instance = ModelFactory.create_model(provider, use_http)  
      
    messages = []  
    if system_prompt:  
        messages.append({"role": "system", "content": system_prompt})  
    messages.extend(history_messages)  
    messages.append({"role": "user", "content": prompt})  
      
    return await model_instance.chat_completion(  
        messages=messages,  
        model=model,  
        temperature=temperature,  
        **kwargs  
    )
