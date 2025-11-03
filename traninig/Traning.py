from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)
from tqdm import tqdm
from transformers.data.data_collator import DataCollator
from datasets import Dataset
from torch.utils.data import IterableDataset
from torch import nn
from comfy.comfy_types import IO
from comfy.utils import ProgressBar
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional,Union
import io
import torch
import os
from pathlib import Path
import json
import folder_paths

output_dir = os.path.join(folder_paths.get_output_directory(),"LLM_Lora")
os.makedirs(output_dir,exist_ok=True)

class Datasets_loader:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        datasets = os.path.join(input_dir,"datasets")
        inputs_types = {
            "required": {
                # ИСПРАВЛЕНО: path_to_datasets на path_to_dataset для единообразия
                "path_to_dataset":("STRING", {"default": datasets}),
                "type_file":(["txt","json"],{"default":"txt"})
            }
        }
        return inputs_types

    RETURN_TYPES = ("DATASETS",)
    RETURN_NAMES = ("datasets",)
    FUNCTION = "load_datasets"
    CATEGORY = "GPT/util/Trainer"
    
    def load_datasets(self,path_to_dataset:str,type_file:str):
        found_files = []
        if not os.path.isdir(path_to_dataset):
            raise FileNotFoundError(f"Directory not found: {path_to_dataset}")
        
        for item_name in os.listdir(path_to_dataset):
            full_path = os.path.join(path_to_dataset, item_name)
            p = Path(full_path)
            # Сравниваем суффикс файла без точки с type_file
            if p.is_file() and p.suffix.lstrip('.') == type_file:
                found_files.append(full_path)
        
        if not found_files:
            raise FileNotFoundError(f"No files of type '{type_file}' found in directory: {path_to_dataset}")
        
        all_data_entries = [] # Этот список будет содержать либо строки для text, либо словари для json
        
        for file_path in found_files:
            with open(file_path, 'r', encoding='utf-8') as f : # Добавил кодировку для надежности
                if type_file == "json":
                    file_content = f.read()
                    try:
                        json_data = json.loads(file_content)
                        # Если JSON-файл содержит список объектов (например, "[{}, {}]")
                        if isinstance(json_data, list):
                            all_data_entries.extend(json_data)
                        # Если JSON-файл содержит один объект (например, "{input:..., output:...}")
                        elif isinstance(json_data, dict):
                            all_data_entries.append(json_data)
                        else:
                            print(f"Warning: Unexpected JSON structure in {file_path}. Expected dict or list of dicts. Skipping.")
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {file_path}: {e}. Skipping file.")
                elif type_file == "txt":
                    # Читаем построчно для текстовых файлов
                    for line in f:
                        stripped_line = line.strip()
                        if stripped_line: # Добавляем только непустые строки
                            all_data_entries.append(stripped_line)
        
        # Теперь создаем Dataset на основе собранных данных
        if type_file == "txt":
            # Для текстовых файлов, каждая запись в all_data_entries - это строка/линия
            datasets = Dataset.from_dict({"text": all_data_entries})
        elif type_file == "json":
            # Для JSON-файлов, all_data_entries - это список словарей
            if not all_data_entries:
                raise ValueError(f"No valid JSON data found in files of type '{type_file}' in directory: {path_to_dataset}")
            
            # Проверяем, что все записи являются словарями, как ожидается в from_list
            if not all(isinstance(entry, dict) for entry in all_data_entries):
                raise TypeError("JSON files must contain either a list of dictionaries or single dictionaries.")
            
            datasets = Dataset.from_list(all_data_entries)
        
        return (datasets,) # Возвращаем кортеж, как ожидается в ComfyUI

class Tokenized_dataset:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs_types = {
            "required": {
                "tokenizer":("TOKENIZER",{}),
                "datasets":("DATASETS",{}),
                # ИСПРАВЛЕНО: Добавлен параметр для указания имени колонки
                "column_name": ("STRING", {"default": "text"}),
                # ИСПРАВЛЕНО: опечатка max_lenght -> max_length
                "max_length":("INT",{"default":1024}),
                "truncation":("BOOLEAN",{"default":True})
            }
        }
        return inputs_types

    RETURN_TYPES = ("TOKENIZED_DATASET",)
    RETURN_NAMES = ("tokenized_dataset",)
    FUNCTION = "create_tokenized_dataset"
    CATEGORY = "GPT/util/Trainer"
    
    def create_tokenized_dataset(
        self,
        tokenizer:PreTrainedTokenizerBase,
        datasets:Dataset,
        column_name:str, # ИСПРАВЛЕНО: добавлен параметр
        max_length:int, # ИСПРАВЛЕНО: опечатка
        truncation:bool
        ):
        
        # ИСПРАВЛЕНО: Проверяем, существует ли указанная колонка в датасете
        if column_name not in datasets.column_names:
            raise ValueError(f"Column '{column_name}' not found in the dataset. Available columns: {datasets.column_names}")

        def tokenize_function(examples):
            return tokenizer(
                # ИСПРАВЛЕНО: Используем переменную column_name вместо жестко заданного "text"
                examples[column_name],
                truncation=truncation,
                max_length=max_length,
                padding="max_length"
            )
        tokenized_dataset = datasets.map(
            tokenize_function,
            batched=True,
            # ИСПРАВЛЕНО: Удаляем ту колонку, которую токенизировали
            remove_columns=[column_name]
        )
        return (tokenized_dataset,)
    
class NodesTrainingArguments:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        output = folder_paths.get_output_directory()
        path = os.path.join(output,"LLM_lora")
        os.makedirs(path,exist_ok=True)
        return {
            "required": {
                "name_file": ("STRING", {"default": path}),
                "overwrite_output_dir": ("BOOLEAN", {"default": False}),
                "do_train": ("BOOLEAN", {"default": True}), # ИСПРАВЛЕНО: По умолчанию должно быть True для обучения
                "do_eval": ("BOOLEAN", {"default": False}),
                "do_predict": ("BOOLEAN", {"default": False}),
                "evaluation_strategy": (["no", "steps", "epoch"], {"default": "no"}),
                "per_device_train_batch_size": ("INT", {"default": 8}),
                "per_device_eval_batch_size": ("INT", {"default": 8}),
                "gradient_accumulation_steps": ("INT", {"default": 1}),
                "learning_rate": ("FLOAT", {"default": 5e-5, "step": 1e-6, "display": "number"}),
                "weight_decay": ("FLOAT", {"default": 0.0}),
                "adam_beta1": ("FLOAT", {"default": 0.9}),
                "adam_beta2": ("FLOAT", {"default": 0.999}),
                "adam_epsilon": ("FLOAT", {"default": 1e-8}),
                "max_grad_norm": ("FLOAT", {"default": 1.0}),
                "num_train_epochs": ("FLOAT", {"default": 3.0}),
                "max_steps": ("INT", {"default": -1}),
                "lr_scheduler_type": (["linear", "cosine", "constant", "constant_with_warmup", "polynomial", "inverse_sqrt"], {"default": "linear"}),
                "warmup_ratio": ("FLOAT", {"default": 0.0}),
                "warmup_steps": ("INT", {"default": 0}),
                "logging_dir": ("STRING", {"default": "./logs"}),
                "logging_strategy": (["no", "steps", "epoch"], {"default": "steps"}),
                "logging_first_step": ("BOOLEAN", {"default": False}),
                "logging_steps": ("INT", {"default": 500}),
                "save_strategy": (["no", "steps", "epoch"], {"default": "steps"}),
                "save_steps": ("INT", {"default": 500}),
                "save_total_limit": ("INT", {"default": -1}), # -1 означает без лимита
                # ИСПРАВЛЕНО: seed должен быть int, а не float
                "seed": ("INT", {"default": 42,"min":0,"max":torch.iinfo(torch.int32).max}),
                "fp16": ("BOOLEAN", {"default": False}),
                "load_best_model_at_end": ("BOOLEAN", {"default": False}),
                "report_to": (["all", "none", "tensorboard", "wandb", "comet_ml"], {"default": "all"}),
                "push_to_hub": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TRAINING_ARGS",)
    RETURN_NAMES = ("training_args",)
    FUNCTION = "training_arguments_config"
    CATEGORY = "GPT/util/Trainer"
    
    def training_arguments_config(self, **kwargs):
        if not kwargs.get("name_file"):
            raise ValueError("File name cannot be empty")
        
        # ИСПРАВЛЕНО: Устанавливаем output_dir и удаляем name_file из kwargs
        output_path = os.path.join(output_dir, kwargs.pop("name_file"))
        kwargs["output_dir"] = output_path

        # ИСПРАВЛЕНО: Убрана опечатка с пробелом в "overwrite_output_dir " и ненужный pop
        
        # Filter out keys that are not part of TrainingArguments
        valid_args = {k: v for k, v in kwargs.items() if k in TrainingArguments.__dataclass_fields__}

        training_args = TrainingArguments(**valid_args)
        return (training_args,)
        

class NodesDataCollatorForLanguageModeling:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs_types = {
            "required": {
                "tokenizer":("TOKENIZER",),
                "mlm":("BOOLEAN",{"default":False}), # ИСПРАВЛЕНО: для GPT-моделей mlm обычно False
            },
            "optional": {
                # ИСПРАВЛЕНО: pad_to_multiple_of и другие необязательные параметры
                "pad_to_multiple_of":("INT", {"default": 8}), # 8 - хорошее значение по умолчанию
            }
        }
        return inputs_types

    RETURN_TYPES = ("DATA_COLLATOR",)
    RETURN_NAMES = ("data_collator",)
    # ИСПРАВЛЕНО: Неправильное имя функции
    FUNCTION = "create_data_collator"
    CATEGORY = "GPT/util/Trainer"
    
    # ИСПРАВЛЕНО: опечатка creatr -> create и kwars_args -> kwargs
    def create_data_collator(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool,
        pad_to_multiple_of: Optional[int] = None
        ):
        
        # ИСПРАВЛЕНО: Правильное создание объекта. tokenizer - обязательный аргумент.
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
            pad_to_multiple_of=pad_to_multiple_of
        )
        return (data_collator,)
    
    
class NodeTrainer:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs_types = {
            "required": {
                # ИСПРАВЛЕНО: опечатка text_mode -> text_model
                "text_model":("TEXT_MODEL",{}),
                "training_args":("TRAINING_ARGS",{}),
                "data_collator":("DATA_COLLATOR",{}), # ИСПРАВЛЕНО: Тип с большой буквы
                "train_dataset":("TOKENIZED_DATASET",{}),
            },
            "optional":{
                # "eval_dataset":("TOKENIZED_DATASET",{}), # Можно добавить в будущем
            }
        }
        return inputs_types
    
    
    RETURN_TYPES = ("TEXT_MODEL",)
    RETURN_NAMES = ("text_model",)
    # ИСПРАВЛЕНО: опечатка traning -> training
    FUNCTION = "training"
    CATEGORY = "GPT/util/Trainer"
    OUTPUT_NODE = True

    def training(
        self,
        text_model: Union[PreTrainedModel, nn.Module],
        training_args: TrainingArguments,
        data_collator: DataCollator,
        train_dataset: Union[Dataset, IterableDataset]
        ):
        
        trainer = Trainer(
            model=text_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
        )
        progressCallback = TqdmProgressCallback()
        trainer.add_callback(progressCallback)
        trainer.train()
        
        # ИСПРАВЛЕНО: trainer.model содержит обученную модель
        trained_model = trainer.model
        trained_model.eval()
        text_model.save_pretrained(training_args.output_dir)
        
        # Сохраняем график в папку temp
        folder_out = folder_paths.get_temp_directory()
        image_graph = progressCallback.plot_training_graph()
        image_path = os.path.join(folder_out, "training_loss_graph.png")
        image_graph.save(image_path)
        
        # Получаем имя файла для превью
        image_filename = os.path.basename(image_path)

        return {
            "ui":{
                "images":[{
                    "filename": image_filename,
                    "subfolder": "",
                    "type": "temp" # ИСПРАВЛЕНО: тип temp для временных файлов
                }]
            },
            "result":(trained_model,)
        }

class TqdmProgressCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.progressBar: Optional[ProgressBar] = None
        self.total_steps = 0
        self.losses = []
        self.steps = []

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps > 0:
            self.total_steps = state.max_steps
            self.pbar = tqdm(total=self.total_steps, unit="step", leave=True)
            self.progressBar = ProgressBar(self.total_steps)
            self.losses = []
            self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pbar and logs:
            loss = logs.get("loss")
            if loss is not None:
                self.pbar.set_postfix({"loss": f"{loss:.4f}"})
                current_step = state.global_step
                self.losses.append(float(loss))
                self.steps.append(current_step)

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar and self.progressBar:
            update_amount = state.global_step - self.pbar.n
            if update_amount > 0:
                self.pbar.update(update_amount)
                self.progressBar.update(update_amount)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()

    def plot_training_graph(self):
        """Генерирует и возвращает изображение диаграммы обучения."""
        if not self.steps or not self.losses:
            # Возвращаем пустое изображение, если нет данных
            return Image.new('RGB', (100, 50), color = 'white')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.steps, self.losses, label='Loss', marker='o', linestyle='-', markersize=4)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Steps')
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf)
        return img