from vllm import LLM, SamplingParams

# Sample prompts.

long_prompt = "The future of AI is ..."
while len(long_prompt) < 1024:
    long_prompt += " " + long_prompt

prompts = [
    long_prompt,
    "Hello, my name is",
    "The president of the United States is The president of the United States is The president of the United States is The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "What is OpenVINO?",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

# Create an LLM.
# llm = LLM(model="facebook/opt-125m")
# llm = LLM(model="/home/sergeys/code/openvino.genai/llm_bench/python/opt-125m/pytorch/dldt/FP16")
# llm = LLM(model="/home/sergeys/code/openvino.genai/llm_bench/python/opt-125m/pytorch/dldt/compressed_weights/OV_FP16-INT4_ASYM")
llm = LLM(model="/home/sergeys/models/new_llms/WW27_llm_2024.3.0-15884-78545386d1a/llama-2-7b-chat/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

