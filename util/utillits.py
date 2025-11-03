from transformers import PreTrainedTokenizerBase
from peft import PeftModel
import torch
import os
import folder_paths

class to:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        inputs_types = {
            "required": {
                "text_model":("TEXT_MODEL",),
                "to":(["cuda","cpu"],{"default":"cuda"}),
                # "dtype":(["float16","float32","None"],{"default":"None"})
            },
        }
        
        return inputs_types
    
    RETURN_TYPES = ("TEXT_MODEL",)
    RETURN_NAMES = ("text_model",)
    FUNCTION = "to"
    CATEGORY = "GPT/util"
    
    def to(
        self,
        text_model,
        to:str,
        # dtype:str | None
        ):
        if hasattr(text_model,"to"):
            if hasattr(text_model, 'peft_config'):
                text_model = text_model.to(to)
                if hasattr(text_model, 'base_model'):
                    text_model.base_model = text_model.base_model.to(to)
            return text_model.to(to)
        raise NameError("Function 'to' Not found")
    
class toAndActivate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs_types = {
            "required": {
                "_":("*"),
                "text_model":("TEXT_MODEL"),
                "to":(["cuda","cpu"],{"default":"cuda"}),
                "dtype":(["float16","float32"],{"default":"float16"})
            }
        }
        
        return inputs_types
    
    RETURN_TYPES = ("TEXT_MODEL",)
    RETURN_NAMES = ("text_model",)
    FUNCTION = "toAndActivate"
    CATEGORY = "GPT/util"
    
    def to(
        self,
        _,
        text_model,
        to:str,
        dtype:str
        ):
        if hasattr(text_model,"to"):
            if hasattr(text_model, 'peft_config'):
                text_model = text_model.to(to)
                if hasattr(text_model, 'base_model'):
                    text_model.base_model = text_model.base_model.to(to)
            return text_model.to(to)
        raise NameError("Function 'to' Not found")
    
    
class Config_MAX_MEMORY:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs_types = {
            "required": {
                "cpu_memory": ("INT",{"default":16,"tooltip":"Сколька выделять память в ГБ,\nпример 10 ГБ"}),
            }
        }
        
        device = torch.cuda.device_count()
        if device <= 0:
            return inputs_types
        # Исправлено: 'default' вместо 'defalt'
        inputs_types["required"]["gpu_select"] = ("INT", {"default": 0, "values": [i for i in range(device)]})
        inputs_types["required"]["gpu_select_memory"] = ("INT", {"default": 8}) # Исправлена опечатка

        return inputs_types

    RETURN_TYPES = ("MAX_MEMORY",)
    RETURN_NAMES = ("max_memory",)
    FUNCTION = "max_memory"
    CATEGORY = "GPT/util"

    def max_memory(
        self,
        cpu_memory: int,
        gpu_select: int | None = None,
        gpu_select_memory: int | None = None
    )->tuple[dict["str","str"]]:
        # {0: "7GiB", "cpu": "16GiB"} - Результат формируется правильно для одного GPU
        max_memory = {
            "cpu": f"{cpu_memory}GiB",
        }
        # Исправлено: проверка на None и добавление записи для выбранного GPU
        if gpu_select is not None and gpu_select_memory is not None:
            # Проверим, что gpu_select не выходит за пределы количества GPU (опционально, но желательно)
            if 0 <= gpu_select < torch.cuda.device_count(): # Проверка на валидность индекса GPU
                 max_memory[gpu_select] = f"{gpu_select_memory}GiB"
            else:
                 print(f"Warning: gpu_select {gpu_select} is out of range (0 to {torch.cuda.device_count()-1}). Skipping GPU memory assignment.")
        # Если gpu_select или gpu_select_memory None, то GPU не добавляется
        return (max_memory,)
    
def filter_response(text):
  start = "<|im_start|>assistant"
  end = "<|im_end|>"
  if start in text and end in text:
    return text.split(start)[1].split(end)[0].strip()
  return ""

class Load_lora_model_peftModel:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        list_paths = folder_paths.get_folder_paths("LLM_lora")
        lora_names = [] # Изменено для хранения только имен
        for i in list_paths:
            if os.path.isdir(i): # Убедимся, что 'i' является директорией
                for f in os.listdir(i):
                    # Добавляем только имя файла/папки, а не полный путь
                    lora_names.append(f)
        
        # Удаляем дубликаты и сортируем для лучшего отображения в UI
        lora_names = sorted(list(set(lora_names)))
                
        inputs_types = {
            "required": {
                "lora": (lora_names,{"default":lora_names[0] if lora_names else "",}), # Используем lora_names и исправляем опечатку "loar"
            }
        }

        return inputs_types

    RETURN_TYPES = ("TEXT_MODEL",)
    RETURN_NAMES = ("text_model",)
    FUNCTION = "load_lora"
    CATEGORY = "GPT/util"
    
    def load_lora(self,model:PreTrainedTokenizerBase,lora:str):
        found_lora_path = None
        # Получаем все настроенные базовые пути для LLM_lora
        for base_path in folder_paths.get_folder_paths("LLM_lora"):
            # Формируем потенциальный полный путь, объединяя базовый путь и имя Lora
            potential_lora_path = os.path.join(base_path, lora)
            
            # Проверяем, существует ли этот путь и является ли он директорией
            # PeftModel обычно ожидает директорию, содержащую adapter_config.json и adapter_model.bin
            if os.path.isdir(potential_lora_path):
                found_lora_path = potential_lora_path
                break # Найдено, нет необходимости искать дальше
        
        if found_lora_path is None:
            raise FileNotFoundError(f"Модель Lora '{lora}' не найдена ни в одном из настроенных путей 'LLM_lora'.")

        if isinstance(model,torch.nn.Module):
            return PeftModel.from_pretrained(model,found_lora_path)
        raise TypeError("Модель должна быть экземпляром torch.nn.Module для загрузки PeftModel.")

