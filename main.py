from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

# Initialize the vLLM model
def load_model(model_name: str):
    return LLM(model=model_name)

# Load model and tokenizer
model_name = "facebook/opt-125m"  # Replace with your model
llm = load_model(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def run_vllm_inference(prompt: str, max_tokens: int = 100):
    # Tokenize input text
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Define sampling parameters
    sampling_params = SamplingParams(max_tokens=max_tokens)
    
    # Run inference
    outputs = llm.generate(input_ids, sampling_params)
    
    # Extract generated token IDs
    generated_token_ids = [output.outputs[0].token_ids for output in outputs]
    
    return generated_token_ids

# Run inference
prompt = "Once upon a time"
outputs = run_vllm_inference(prompt)
print(outputs)
