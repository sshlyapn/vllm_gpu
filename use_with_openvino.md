# Use vLLM with OpenVINO

## Build Docker Image

```bash
git clone --branch openvino-model-executor https://github.com/ilya-lavrenov/vllm.git
cd vllm
docker build -t vllm:openvino -f Dockerfile.openvino .
```

Once it successfully finishes you will have a `vllm:openvino` image. It can directly spawn a serving container with OpenAI API endpoint or you can work with it interactively via bash shell.

## Use vLLM serving with OpenAI API

_All below steps assume you are in `vllm` root directory._

### Start The Server:

```bash
# It's advised to mount host HuggingFace cache to reuse downloaded models between the runs.
docker run --rm -p 8000:8000 -v $HOME/.cache/huggingface:/root/.cache/huggingface vllm:openvino --model meta-llama/Llama-2-7b-hf --port 8000 --disable-log-requests --swap-space 50

### Additional server start up parameters that could be useful:
# --max-num-seqs <max number of sequences per iteration> (default: 256)
# --swap-space <GiB for KV cache> (default: 4)
```

### Request Completion With Curl:

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"meta-llama/Llama-2-7b-hf", "prompt": "What is the key advantage of Openvino framework","max_tokens": 300, "temperature": 0.7}'
```

### Run Benchmark

Let's run [benchmark_serving.py](https://github.com/ilya-lavrenov/vllm/blob/openvino-model-executor/benchmarks/benchmark_serving.py):

```bash
cd benchmarks
# Download dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Install PyPI dependencies (vLLM and aiohttp seem to be enough)
pip3 install vllm==0.3.3 aiohttp

# Launch benchmark script
python3 benchmark_serving.py --backend openai --endpoint /v1/completions --port 8000 --model meta-llama/Llama-2-7b-hf --dataset ShareGPT_V3_unfiltered_cleaned_split.json

### Additional benchmark_serving.py parameters that could be useful:
# --num-prompts <number of requests to send> (default: 1000)
# --request-rate <requests sent per second> (default: "inf" - special value "inf" means we send all n requests at once)
```


## Use vLLM offline 

_All below steps assume you are in `vllm` root directory._

The `vllm:openvino` image does not contain any samples by default, but since you have a vLLM repository cloned you can mount it to a container and use the samples from vLLM repository from the inside of the running container

### Run Example

Let's run [offline_inference.py](https://github.com/ilya-lavrenov/vllm/blob/openvino-model-executor/examples/offline_inference.py):

```bash
# It's advised to mount host HuggingFace cache to reuse downloaded models between the runs.
docker run --rm -it --entrypoint python3 -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $PWD:/workspace/vllm vllm:openvino /workspace/vllm/examples/offline_inference.py
```

### Run Benchmark

You can also run offline benchmark. Let's run [benchmark_throughput.py](https://github.com/ilya-lavrenov/vllm/blob/openvino-model-executor/benchmarks/benchmark_throughput.py):

```bash
# Download the dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# It's advised to mount host HuggingFace cache to reuse downloaded models between the runs.
docker run --rm -it --entrypoint python3 -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $PWD:/workspace/vllm vllm:openvino /workspace/vllm/benchmarks/benchmark_throughput.py --model meta-llama/Llama-2-7b-hf --dataset /workspace/vllm/ShareGPT_V3_unfiltered_cleaned_split.json --device auto

### Additional benchmark_throughput.py parameters that could be useful:
# --num-prompts <number of requests to send> (default: 1000)
# --swap-space <GiB for KV cache> (default: 50)
```
