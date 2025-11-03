from .Load_GPT import LoadAutoModel,LoadAutoTokenizer
from .Generate.Generate import GPTMulityGenerate,GPTTextGenerator
from .util.utillits import to,toAndActivate,Config_MAX_MEMORY
from .traninig.Traning import Tokenized_dataset,Datasets_loader,NodeTrainer,NodesTrainingArguments,NodesDataCollatorForLanguageModeling
import folder_paths
import os
from pathlib import Path


comfy_llm_folder = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(comfy_llm_folder, exist_ok=True)
folder_paths.add_model_folder_path("LLM", comfy_llm_folder)

comfy_llm_folder = os.path.join(folder_paths.models_dir, "LLM_lora")
os.makedirs(comfy_llm_folder, exist_ok=True)
folder_paths.add_model_folder_path("LLM_lora", comfy_llm_folder)

NODE_CLASS_MAPPINGS = {
    "LoadAutoTokenizer":LoadAutoTokenizer.LoadAutoTokenizer,
    "LoadAutoModel":LoadAutoModel.LoadAutoModel,
    "GPTTextGenerator": GPTTextGenerator,
    "GPTMulityGenerate":GPTMulityGenerate,
    "to": to,
    "toAndActivate":toAndActivate,
    "Config_MAX_MEMORY":Config_MAX_MEMORY,
    
    "Tokenized_dataset":Tokenized_dataset,
    "Datasets_loader":Datasets_loader,
    "NodeTrainer":NodeTrainer,
    "NodesTrainingArguments":NodesTrainingArguments,
    "NodesDataCollatorForLanguageModeling":NodesDataCollatorForLanguageModeling
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAutoTokenizer":"Load Tokenizer",
    "LoadAutoModel":"Load Text Model",
    "GPTTextGenerator": "GPT Text Generator",
    "GPTMulityGenerate":"GPT Mulity Generate",
    "to": "to model",
    "toAndActivate":"to model and or activate",
    "Config_MAX_MEMORY": "Configure max memory",

    "Tokenized_dataset":"Tokenized_dataset",
    "Datasets_loader":"Datasets_loader",
    "NodeTrainer":"NodeTrainer",
    "NodesTrainingArguments":"NodesTrainingArguments",
    "NodesDataCollatorForLanguageModeling":"NodesDataCollatorForLanguageModeling"
}



__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']