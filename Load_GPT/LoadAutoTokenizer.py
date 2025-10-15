from transformers import (
    # Авто-классы
    AutoTokenizer,
    AutoProcessor,
    AutoFeatureExtractor,
    AutoImageProcessor,

    # Специализированные токенайзеры
    GPT2Tokenizer, GPT2TokenizerFast,

    # Мультимодальные и специальные
    CLIPTokenizer, CLIPTokenizerFast,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    VisionTextDualEncoderProcessor,
)

import os
import folder_paths
from typing import Any


tokenizer_classes_dict = [
    {"AutoTokenizer": AutoTokenizer},
    {"AutoProcessor": AutoProcessor},
    {"AutoFeatureExtractor": AutoFeatureExtractor},
    {"AutoImageProcessor": AutoImageProcessor},
    {"GPT2Tokenizer": GPT2Tokenizer},
    {"GPT2TokenizerFast": GPT2TokenizerFast},
    {"CLIPTokenizer": CLIPTokenizer},
    {"CLIPTokenizerFast": CLIPTokenizerFast},
    {"VisionTextDualEncoderProcessor": VisionTextDualEncoderProcessor}
]


class LoadAutoTokenizer:
    def __init__(self) -> None:
        self.tokenizer = None
        
    @classmethod
    def INPUT_TYPES(cls):
        llm_path = os.path.join(folder_paths.models_dir, "LLM")
       
        if not os.path.exists(llm_path):
            os.makedirs(llm_path)
            
        available_models = []
        if os.path.exists(llm_path):
            for item in os.listdir(llm_path):
                item_path = os.path.join(llm_path, item)
                if os.path.isdir(item_path):
                    config_path = os.path.join(item_path, "config.json")
                    if os.path.exists(config_path):
                        available_models.append(item)
        key_tokenizer = [key for dictionary in tokenizer_classes_dict for key in dictionary.keys()]
        return {
            "required": {
                "tokenizer_model": (available_models, {"default": available_models[0] if available_models else ""}),
                "type_class": (key_tokenizer,{"default":"AutoTokenizer"})
            }
        }
    
    RETURN_TYPES = ("TOKENIZER",)
    RETURN_NAMES = ("tokenizer",)
    FUNCTION = "load_AutoTokeinzer"
    CATEGORY = "GPT/Loaders"
    
    def load_AutoTokeinzer(self,tokenizer_model,type_class:Any):
        model_path = os.path.join(folder_paths.models_dir, "LLM", tokenizer_model)
        if not os.path.exists(model_path) :
            raise FileNotFoundError("If the path is not provided or the model is not found.")
        tokenizer_class = None
        for i in tokenizer_classes_dict:
             if type_class in i:
                tokenizer_class = i[type_class]
                break
        if tokenizer_class is None:
            raise Exception(f"Class not found for name: {type_class}")
        print(f"Type Tokenizer : {type(tokenizer_class)}\nLoad model : {tokenizer_model}")
        self.tokenizer = tokenizer_class.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,offline=True)
        print(f"Load Done {tokenizer_model}")
        if not (isinstance(self.tokenizer,PreTrainedTokenizerBase) or isinstance(self.tokenizer,ProcessorMixin)):
            raise TypeError("Loaded model is not an instance of PreTrainedTokenizerBase or ProcessorMixin.")
        return (self.tokenizer,)
    