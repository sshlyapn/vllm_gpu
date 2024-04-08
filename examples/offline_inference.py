from vllm import LLM, SamplingParams
import torch

# Sample prompts.
prompts = [
    "What is OpenVINO?",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=42, max_tokens=30)

model_id = "facebook/opt-125m"

# Create an LLM.
llm = LLM(model=model_id, device="auto", kv_cache_dtype="u8")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

###########################
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_id)
ov_model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer.encode(prompts[0], return_tensors='pt')
output = ov_model.generate(inputs, max_new_tokens=30, do_sample=True, temperature=0.8, top_p=0.95)

text = tokenizer.batch_decode(output)
print(text)
