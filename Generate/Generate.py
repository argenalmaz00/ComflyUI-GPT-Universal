import gc
import torch
from typing import Any
from tqdm import tqdm
from comfy.utils import ProgressBar
import latent_preview
from transformers import BatchEncoding, TextStreamer, PreTrainedTokenizerBase, ProcessorMixin,BatchFeature,GenerationMixin,PreTrainedModel,AutoImageProcessor
from jinja2.exceptions import TemplateError


class ProgressStreamer (TextStreamer):
    def __init__(self, tokenizer, total_tokens=100, **kwargs):
        super().__init__(tokenizer)
        self.progressBar = ProgressBar(total=total_tokens)
        self.tokenizer = tokenizer
        self.onStream = False
        
    def put(self, value):
        # Обновляем прогресс-бар
        self.progressBar.update(1)
        typing = self.tokenizer.decode(value[0],skip_special_tokens=True)
        print(typing)
        print(value, end="", flush=True)
    
    def end(self):
        pass


class GPTTextGenerator:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs_types = {
            "required": {
                "system_role": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "user_role": ("STRING", {"multiline": True, "default": "Hello, how are you?"}),
                "text_model": ("TEXT_MODEL",),
                "tokenizer": ("TOKENIZER",),
                # "device": (["cuda", "cpu","auto"], {"default": "auto"}),
                # "torch_dtype": (["float16", "float32"], {"default": "float16"}),
                "type_message": (["raw_text","apply_chat_template"], {"default": "apply_chat_template"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
                "max_length": ("INT", {"default": 512, "min": 10, "max": 1024}),
                "max_new_length":("INT",{"default":"512"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": torch.iinfo(torch.int32).max}),
                "enable_text_stream":("BOOLEAN",{"default":False}),
            },
            "optional":{
                "image": ("IMAGE",)
            }
        }
        
        return inputs_types

    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "generate"
    CATEGORY = "GPT"
    
    def generate(
        self,
        system_role: str,
        user_role: str,
        text_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | ProcessorMixin,
        # device,
        # torch_dtype,
        type_message:str,
        temperature:int,
        top_k:int,
        top_p:int,
        max_length:int,
        max_new_length:int,
        seed:int,
        enable_text_stream:bool,
        image=None
        )-> tuple[str]:
        
        torch.manual_seed(seed)
        if torch.cuda.is_available() and text_model.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu" if device == "cpu" else text_model.device
        # dtype = torch.float16 if torch_dtype == "float16" else torch.float32
        
        # if hasattr(text_model, 'peft_config'):
        #     text_model = text_model.to(target_device, dtype=dtype)
        #     if hasattr(text_model, 'base_model'):
        #         text_model.base_model = text_model.base_model.to(target_device, dtype=dtype)
        # else:
        #     text_model = text_model.to(target_device, dtype=dtype)
        if type_message == "apply_chat_template":
            message = [
                {"role": "system", "content":system_role},
                {"role": "user", "content":user_role}
            ]
            if isinstance(tokenizer,PreTrainedTokenizerBase):
                if hasattr(tokenizer,"apply_chat_template") and hasattr(tokenizer,"chat_template"):
                    try :
                        text = tokenizer.apply_chat_template(message,add_generation_prompt=True,tokenize=False)
                        inputs = tokenizer(text=text,return_tensors="pt")
                    except TemplateError as e:
                        if e.message == "Conversation roles must alternate user/assistant/user/assistant/...":
                            for i,m in enumerate (message):
                                if m["role"] == "system":
                                    message.remove(m)
                                    print(f"Remove system prompt: {m["content"]}")
                                    
                            text = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
                            inputs = tokenizer(text=text,return_tensors="pt")
                        else :
                            raise e
                else :
                    inputs = tokenizer(user_role,return_tensors="pt")
                    
            elif isinstance(tokenizer,ProcessorMixin) or isinstance(tokenizer,AutoImageProcessor):
                text = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True) # type: ignore
                if image is not None:
                    inputs = tokenizer(images=image,text=text,return_tensors="pt") # type: ignore
                else:
                    inputs = tokenizer(text=text,return_tensors="pt") # type: ignore
            else :
                raise Exception(f"Calss Not code: ${type(tokenizer)}")
        else :
            raw_text = f"""<|im_start|>system
                {system_role}<|im_end|>
                <|im_start|>user
                {user_role}<|im_end|>
                <|im_start|>assistant
            """
            inputs = tokenizer(raw_text,return_tensors="pt",add_special_tokens=False)
            
        if inputs is None:
            print(inputs)
            raise Exception("Error inputs_ids Empty")
        
        stream_text:ProgressStreamer | None = None
        try:
            if enable_text_stream :
                stream_text = ProgressStreamer(tokenizer)
        except Exception:
            print("Unknown Error stream,Skip stream")
        response = ""
        try:
            with torch.no_grad():
                # inputs.to(text_model.device)
                if isinstance(text_model,GenerationMixin):           
                    inputs = inputs.to(text_model.device)
                    print(inputs)
                    print("==" * 10)
                    print(f"is Tensor input_ids : {hasattr(inputs,"input_ids") } and Type (${type(inputs['input_ids'] if hasattr(inputs,"input_ids") else None)})")
                    print(f"is Tensor attention_mask : {hasattr(inputs,"attention_mask") } and Type (${type(inputs['attention_mask'] if hasattr(inputs,"attention_mask") else None)})")
                    print("==" * 10)
                    
                    length = len(inputs["input_ids"][0]) + max_length
                    generated_ids = text_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=length,
                        # max_new_tokens=max_new_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=True,
                        repetition_penalty=1.1,
                        pad_token_id=(tokenizer.eos_token_id if hasattr(tokenizer,"eos_token_id") else None ), # type: ignore
                        eos_token_id=(tokenizer.eos_token_id if hasattr(tokenizer,"eos_token_id") else None ), # type: ignore
                        streamer=stream_text if enable_text_stream else None
                    )
                                                
                    response = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
                else :
                    raise Exception(f"Error not Class GenerateionMixin (${isinstance(text_model,GenerationMixin)}) : ${type(text_model)}")
                
                if "<|im_start|>assistant\n" in response:
                    response = response.split("<|im_start|>assistant\n")[1]
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]
        
        finally:
            if hasattr(text_model,"device"):
                if text_model.device != "cpu":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        print("=" * 30 + "\n" + response.strip() + "\n" + "=" * 30)
        return (response.strip(),)