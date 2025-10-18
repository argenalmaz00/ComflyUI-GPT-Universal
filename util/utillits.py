from transformers import PreTrainedTokenizerBase
import torch

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