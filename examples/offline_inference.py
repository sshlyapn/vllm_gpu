from vllm import LLM, SamplingParams
import torch

# Sample prompts.
prompts = [
    "What is OpenVINO?",
    # "How to cook beef steak in the oven? What is the required temperature?",
    # "Do you know What is OpenVINO?",
]
prompts = [
    "What is OpenVINO?",
    "What is OpenVINO?",
    "What is OpenVINO?",
    "What is OpenVINO?",

    "What is OpenVINO?",
    "What is OpenVINO?",
    "What is OpenVINO?",
    "What is OpenVINO?",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512, ignore_eos=True, seed=42)

# Create an LLM.
llm = LLM(model="mistralai/Mistral-7B-v0.1", device="auto", max_model_len=16384)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

###########################
# from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# set_seed(42)

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
# ov_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# inputs = tokenizer.encode(prompts[0], return_tensors='pt')
# output = ov_model.generate(inputs, max_new_tokens=30, do_sample=True, temperature=0.8, top_p=0.95)

# text = tokenizer.batch_decode(output)
# print(text)
