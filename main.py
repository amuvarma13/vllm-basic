import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
# Ensure you've installed the SNAC library (or provide the correct import here)
# e.g. `pip install git+https://github.com/hubertsiuzdak/snac.git`
from snac import SNAC

# Load the SNAC model for decoding the audio codes
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

# Orpheus TTS model info
model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
engine = LLM(model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Common token IDs for Orpheus TTS
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]
STOP_TOKEN_ID = 128258
AUDIO_MARKER_TOKEN = 128257  # e.g. <custom_token_4> or whichever signals the audio start

def parse_tokens_to_audio(generated_ids, snac_model):
    """
    Takes the raw token IDs generated by the model, extracts & processes the
    audio codes, and decodes them into an audio waveform using snac_model.
    Returns a list of decoded audio samples (tensors).
    """
    # Find the last occurrence of AUDIO_MARKER_TOKEN and crop everything before it
    token_indices = (generated_ids == AUDIO_MARKER_TOKEN).nonzero(as_tuple=True)
    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx + 1:]
    else:
        cropped_tensor = generated_ids

    # Remove the STOP_TOKEN_ID
    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != STOP_TOKEN_ID]
        processed_rows.append(masked_row)

    # Group every 7 tokens and subtract offsets
    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)

    # Helper function to re-map codes into separate layers for SNAC
    def redistribute_codes(code_list):
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list) + 1) // 7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i + 1] - 4096)
            layer_3.append(code_list[7*i + 2] - (2 * 4096))
            layer_3.append(code_list[7*i + 3] - (3 * 4096))
            layer_2.append(code_list[7*i + 4] - (4 * 4096))
            layer_3.append(code_list[7*i + 5] - (5 * 4096))
            layer_3.append(code_list[7*i + 6] - (6 * 4096))

        # Convert to tensors for the snac_model decoder
        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0)
        ]
        # Decode to produce final audio
        audio_hat = snac_model.decode(codes)
        return audio_hat

    # Decode each code sequence to audio
    audio_samples = []
    for code_list in code_lists:
        samples = redistribute_codes(code_list)
        audio_samples.append(samples)

    return audio_samples

def text_to_speech(prompt, voice=None):
    """
    Given a text prompt and optional voice, generates audio tokens using
    the Orpheus TTS model and decodes them into audio samples via SNAC.
    """

    # Construct the prompt with optional voice prefix
    if voice:
        adapted_prompt = f"{voice}: {prompt}"
        prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
    else:
        prompt_tokens = tokenizer(prompt, return_tensors="pt")

    # Insert special tokens
    start_token = torch.tensor([[START_TOKEN_ID]], dtype=torch.int64)
    end_tokens = torch.tensor([END_TOKEN_IDS], dtype=torch.int64)
    all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)

    # Decode to string for LLM
    final_prompt = tokenizer.decode(all_input_ids[0])

    # Sampling parameters
    params = SamplingParams(
        temperature=0.6,
        top_p=0.8,
        max_tokens=1200,
        stop_token_ids=[STOP_TOKEN_ID],
        repetition_penalty=1.3
    )

    # Generate token IDs from the model
    outputs = engine.generate([final_prompt], [params])
    token_ids = outputs[0].outputs[0].token_ids
    generated_ids = torch.tensor([token_ids], dtype=torch.long)

    # Convert generated tokens into audio
    audio_samples = parse_tokens_to_audio(generated_ids, snac_model)
    return audio_samples


    # Example usage
audio_output = text_to_speech("Hello world", voice="zoe")
print("Decoded audio (tensors):", audio_output)
