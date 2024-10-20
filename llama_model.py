
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from huggingface_hub import login
from peft import prepare_model_for_kbit_training
from peft import PeftModel
import torch
import transformers
# from langchain.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline

login("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")

def Load_llama_model():
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model=PeftModel.from_pretrained(base_model, "101a44ad-4c86-4b89-bed7-bf555d08bb50_adaptive_layer/peft_model")
    model=model.merge_and_unload()
    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=300,
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm