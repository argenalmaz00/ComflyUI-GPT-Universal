import torch
import comfy.model_management as model_management
import comfy.utils
import os
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoProcessor
from peft import PeftModel
import folder_paths
import base64
from PIL import Image
import io

class GPTTextGenerator:
    def __init__(self):
        pass
   
    @classmethod
    def INPUT_TYPES(cls):
        inputs_types = {
            "required": {
                "system_role": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "user_role": ("STRING", {"multiline": True, "default": "Hello, how are you?"}),
                "model": ("MODEL",),
                "tokenizer": ("TOKENIZER",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "torch_dtype": (["float16", "float32"], {"default": "float16"}),
                "apply_chat_template": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
                "max_length": ("INT", {"default": 512, "min": 10, "max": 2048}),
                "seed": ("INT", {"default": 42, "min": 0, "max": torch.iinfo(torch.int32).max}),
            }
        }
        
        return inputs_types
   
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_response"
    CATEGORY = "GPT"

    def generate_response(self, system_role, user_role, model, tokenizer, device, torch_dtype,apply_chat_template,temperature, top_k, top_p, max_length, seed):

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch_dtype == "float16" else torch.float32
        
        if hasattr(model, 'peft_config'):
            model = model.to(target_device, dtype=dtype)
            if hasattr(model, 'base_model'):
                model.base_model = model.base_model.to(target_device, dtype=dtype)
        else:
            model = model.to(target_device, dtype=dtype)
        


        raw_text = f"""<|im_start|>system
            {system_role}<|im_end|>
            <|im_start|>user
            {user_role}<|im_end|>
            <|im_start|>assistant
        """
        inputs = tokenizer(raw_text,return_tensors="pt",add_special_tokens=False).to(target_device)
    
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=len(inputs["input_ids"][0]) + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                response = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
                
                if "<|im_start|>assistant\n" in response:
                    response = response.split("<|im_start|>assistant\n")[1]
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]
        
        finally:
            if hasattr(model, 'peft_config'):
                model = model.to("cpu")
                if hasattr(model, 'base_model'):
                    model.base_model = model.base_model.to("cpu")
            else:
                model = model.to("cpu")
            
            if target_device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        return (response.strip(),)

    
    
class LoadGPTModel:
    
   def __init__(self):
       pass
   
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
       
       if not available_models:
           available_models = ["No models found - place models in models/LLM folder"]
       
       return {
           "required": {
               "model": (available_models, {"default": available_models[0] if available_models else ""}),
           }
       }
   
   RETURN_TYPES = ("MODEL", "TOKENIZER")
   RETURN_NAMES = ("model", "tokenizer")
   FUNCTION = "load_model"
   CATEGORY = "GPT/Loaders"

   def load_model(self, model):
        if "No models found" in model:
            error_msg = "No models found in models/LLM folder. Please place your model there."
            raise Exception(error_msg)
        
        model_path = os.path.join(folder_paths.models_dir, "LLM", model)
        
        if not os.path.exists(model_path):
            error_msg = f"Model path does not exist: {model_path}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        print(f"Loading model from: {model_path}")
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            offline=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Loading model...")
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print(f"Model loaded successfully: {model}")
        return (model_obj, tokenizer)
           
class GPTLoadLora:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        llm_lora_path = os.path.join(folder_paths.models_dir, "LLM_lora")

        if not os.path.exists(llm_lora_path):
            os.makedirs(llm_lora_path)

        available_loras = []
        if os.path.exists(llm_lora_path):
            for item in os.listdir(llm_lora_path):
                item_path = os.path.join(llm_lora_path, item)
                if os.path.isdir(item_path):
                    adapter_config = os.path.join(item_path, "adapter_config.json")
                    adapter_weights = os.path.join(item_path, "adapter_model.safetensors")
                    if os.path.exists(adapter_config) and os.path.exists(adapter_weights):
                        available_loras.append(item)

        if not available_loras:
            available_loras = ["No LoRA adapters found - place LoRA in models/LLM_lora folder"]
        return {
           "required": {
               "model": ("MODEL",),
               "lora": (available_loras, {"default": available_loras[0] if available_loras else ""}),
           }
       }

    RETURN_TYPES = ("MODEL", )
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "GPT/Loaders"
    
    def load_lora(self, model, lora):
        if "No LoRA adapters found" in lora:
            error_msg = "No LoRA adapters found in models/LLM_lora folder. Please place your LoRA there."
            raise Exception(error_msg)
        
        lora_path = os.path.join(folder_paths.models_dir, "LLM_lora", lora)
        
        if not os.path.exists(lora_path):
            error_msg = f"LoRA path does not exist: {lora_path}"
            raise FileNotFoundError(error_msg)
        
        print(f"Loading LoRA from: {lora_path}")
        
        model_with_lora = PeftModel.from_pretrained(
            model, 
            lora_path,
            local_files_only=True,
            trust_remote_code=False
        )
        
        print(f"LoRA loaded successfully: {lora}")
        return (model_with_lora,)

class LoadGPTVison:
    def __init__(self) -> None:
        pass
    

NODE_CLASS_MAPPINGS = {
   "LoadGPTModel": LoadGPTModel,
   "GPTLoadLora": GPTLoadLora,
   "GPTTextGenerator": GPTTextGenerator,
   "LoadGPTVison":LoadGPTVison
}

NODE_DISPLAY_NAME_MAPPINGS = {
   "LoadGPTModel": "Load GPT Model",
   "GPTLoadLora": "GPT Load Lora", 
   "GPTTextGenerator": "GPT Text Generator"
}