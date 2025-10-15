# from .gptchat import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .Load_GPT import LoadAutoModel,LoadAutoTokenizer
from .Generate import Generate
from .util.utillits import to,toAndActivate,Config_MAX_MEMORY

# Регистрация нод
NODE_CLASS_MAPPINGS = {
    "LoadAutoTokenizer":LoadAutoTokenizer.LoadAutoTokenizer,
    "LoadAutoModel":LoadAutoModel.LoadAutoModel,
    "GPTTextGenerator": Generate.GPTTextGenerator,
    "to": to,
    "toAndActivate":toAndActivate,
    "Config_MAX_MEMORY":Config_MAX_MEMORY
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAutoTokenizer":"Load Tokenizer",
    "LoadAutoModel":"Load Text Model",
    "GPTTextGenerator": "GPT Text Generator",
    "to": "to model",
    "toAndActivate":"to model and or activate",
    "Config_MAX_MEMORY": "Configure max memory"
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']