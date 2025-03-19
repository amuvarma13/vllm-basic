import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Set up model parameters
model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
dtype = torch.bfloat16

# Create engine and tokenizer
engine = LLM(model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example prompt and optional voice
prompt = "Hello world"
voice = "zoe"  # Set to None or empty string if you don't want to use a voice

# Build the prompt string, adding voice tokens if provided
if voice:
    # Add voice prefix
    adapted_prompt = f"{voice}: {prompt}"
    prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
else:
    # Just encode the raw prompt
    prompt_tokens = tokenizer(prompt, return_tensors="pt")

# Start/end tokens for Orpheus TTS
start_token = torch.tensor([[128259]], dtype=torch.int64)
end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)

# Convert the entire sequence of IDs to a decoded string
final_prompt = tokenizer.decode(all_input_ids[0])

# Sampling parameters
params = SamplingParams(
    temperature=0.6,
    top_p=0.8,
    max_tokens=1200,
    stop_token_ids=[49158],
    repetition_penalty=1.3
)

# Generate tokens
outputs = engine.generate([final_prompt], [params])

# Extract the token IDs from the model output
token_ids = [tok.token_id for tok in outputs[0].outputs[0].tokens]

print("Generated token IDs:", token_ids)
