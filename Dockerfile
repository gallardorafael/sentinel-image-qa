# Dockerfile to use the vLLM container as base to build the Sentinel container
# this is temporal until Qwen2-VL ls ready for OpenAI compatible serving
# Check Note #4 here: https://github.com/vllm-project/vllm/pull/7905
# Once Qwen2-VL is ready, I will split the Sentinel container from the vLLM container
# so we have a clear client (Sentinel Streamlit App) and server (vLLM with Qwen2-VL OpenAI API) separation
# with intented usage similar to: https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html

FROM vllm/vllm-openai:v0.6.2

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sentinel-workspace
COPY . /sentinel-workspace

RUN pip install --upgrade pip

RUN pip3 install streamlit streamlit-cropper qwen-vl-utils bitsandbytes>0.37.0

ENTRYPOINT []