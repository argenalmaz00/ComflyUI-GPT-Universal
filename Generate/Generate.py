import torch
from tqdm import tqdm
from comfy.utils import ProgressBar
from transformers import (
    BatchEncoding,
    TextStreamer,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    BatchFeature,
    GenerationMixin,
    PreTrainedModel,
    AutoImageProcessor,
    StoppingCriteria, 
    StoppingCriteriaList,
)
from jinja2.exceptions import TemplateError
from ..util.utillits import filter_response

class TokenLengthStoppingCriteria(StoppingCriteria):
    def __init__(self, input_length, max_new_tokens):
        self.input_length = input_length
        self.max_new_tokens = max_new_tokens
        self.tqdm = tqdm(total=max_new_tokens, unit_scale=True)
        self.progressBar = ProgressBar(total=max_new_tokens)
        self.last_current_new_tokens = 0

    def __call__(self, input_ids, scores, **kwargs):
        current_new_tokens = input_ids.shape[1] - self.input_length
        
        increment = current_new_tokens - self.last_current_new_tokens
        if increment > 0:
            self.tqdm.update(increment)
            self.progressBar.update(increment)
        
        self.last_current_new_tokens = current_new_tokens
        
        if current_new_tokens >= self.max_new_tokens:
            self.tqdm.close()
            return True
        return False

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
                "type_message": (["raw_text","apply_chat_template"], {"default": "apply_chat_template"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
                "max_new_tokens":("INT",{"default": 512, "min": 10, "max": 1024}),
                "seed": ("INT", {"default": 42, "min": 0, "max": torch.iinfo(torch.int32).max}),
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
        temperature:float,
        top_k:int,
        top_p:int,
        max_new_tokens:int,
        seed:int,
        image=None
        )-> tuple[str]:
        
        torch.manual_seed(seed)
        if torch.cuda.is_available() and text_model.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        message = [
            {"role": "system", "content":system_role},
            {"role": "user", "content":user_role}
        ]
        if isinstance(tokenizer,PreTrainedTokenizerBase):
            if image:
                print("Tokenizer Image dose not support")
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
            
        if inputs is None:
            print(inputs)
            raise Exception("Error inputs_ids Empty")
        response = ""
        try:
            with torch.no_grad():
                if isinstance(text_model,GenerationMixin):
                    inputs = inputs.to(text_model.device)
                    user_token_len = len(inputs["input_ids"][0])
                    tokenLengthStoppingCriteria = TokenLengthStoppingCriteria(user_token_len,max_new_tokens=max_new_tokens)
                    generated_ids = text_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=True,
                        repetition_penalty=1.1,
                        pad_token_id=(tokenizer.eos_token_id if hasattr(tokenizer,"eos_token_id") else None ), # type: ignore
                        eos_token_id=(tokenizer.eos_token_id if hasattr(tokenizer,"eos_token_id") else None ), # type: ignore
                        stopping_criteria=StoppingCriteriaList([tokenLengthStoppingCriteria])
                    )
                    response = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
                    response = filter_response(response)
                else :
                    raise Exception(f"Error not Class GenerateionMixin (${isinstance(text_model,GenerationMixin)}) : ${type(text_model)}")
        finally:
            if hasattr(text_model,"device"):
                if text_model.device != "cpu":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        print("=" * 30 + "\n" + response.strip() + "\n" + "=" * 30)
        return (response.strip(),)