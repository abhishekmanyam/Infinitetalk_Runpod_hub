# Self-contained runtime image. Models are downloaded in parallel inside the
# build (download_models.py) so we stay under RunPod's 30-min build cap.
# Set VARIANT=single (default) or multi to choose the InfiniteTalk weight.

FROM wlsdml1114/engui_genai-base:1.9 AS runtime

ARG VARIANT=single
ENV VARIANT=${VARIANT}
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]" && \
    pip install runpod websocket-client librosa sageattention

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    pip install -r ComfyUI-Manager/requirements.txt && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    pip install -r ComfyUI-GGUF/requirements.txt && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    pip install -r ComfyUI-KJNodes/requirements.txt && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    pip install -r ComfyUI-VideoHelperSuite/requirements.txt && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer && \
    pip install -r ComfyUI-MelBandRoFormer/requirements.txt && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    pip install -r ComfyUI-WanVideoWrapper/requirements.txt

# Parallel model download in a single layer.
COPY download_models.py /tmp/download_models.py
RUN MODELS_DIR=/ComfyUI/models python /tmp/download_models.py && \
    rm -rf /tmp/download_models.py /root/.cache/huggingface

COPY . .
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
