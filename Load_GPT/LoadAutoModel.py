import os
import folder_paths
from typing import Any

from transformers import (
    # Auto-models
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    PreTrainedModel,
)

auto_model_classes_dict = [
    {"AutoModel": AutoModel},
    {"AutoModelForCausalLM": AutoModelForCausalLM},
    {"AutoModelForVision2Seq": AutoModelForVision2Seq}
]



class LoadAutoModel:
    def __init__(self) -> None:
        self.text_model = None
        
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
        key_auto_model_classes = [key for dictionary in auto_model_classes_dict for key in dictionary.keys()]
        return {
            "required": {
                "model": (available_models, {"default": available_models[0] if available_models else ""}),
                "type_class": (key_auto_model_classes,{"default":"AutoModel"}),
                "torch_dtype" : (["auto","float32","float16","bfloat16"],{"default":"auto"}),
                "device_map" : (["auto","cpu","cuda"],{"default":"auto"}),
                "load_in_8bit":("BOOLEAN",{"default":False})
            },
            "optional":{
                "max_memory": ("MAX_MEMORY",),
            }
        }
    
    # Correct tuple forms for ComfyUI and function mapping
    RETURN_TYPES = ("TEXT_MODEL",)
    RETURN_NAMES = ("text_model",)
    FUNCTION = "load_AutoModel"
    CATEGORY = "GPT/Loaders"
    
    def load_AutoModel(self, model,type_class:Any,torch_dtype:str,device_map:str,load_in_8bit:bool,max_memory:str | None = None):
        model_path = os.path.join(folder_paths.models_dir, "LLM", model)
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model path not found; ensure model files exist locally.")
        
        # type_class is a string name chosen from the UI; find corresponding class
        model_class = None
        for mapping in auto_model_classes_dict:
            if type_class in mapping:
                model_class = mapping[type_class]
                break
        if model_class is None:
            raise Exception(f"Class not found for name: {type_class}")

        print(f"Model class: {model_class}  | Load model path: {model_path}")
        print(f"Attempting to load model from: {model_path} with class: {model_class.__name__}")
        try:
            # load using the found class; ensure local-only to avoid downloads
            self.text_model = model_class.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
                force_download=False,
                resume_download=False,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_8bit=load_in_8bit,
                max_memory=max_memory
            )
            print(f"Successfully loaded model from: {model_path}")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            raise # Re-raise the exception after logging it
        if not isinstance(self.text_model, PreTrainedModel):
            raise TypeError("Loaded model is not an instance of PreTrainedModel.")
        return (self.text_model,)
