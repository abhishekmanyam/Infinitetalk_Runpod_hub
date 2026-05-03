# Runtime image. Models come from a prebuilt models image (see models.Dockerfile)
# so this Dockerfile rebuilds in minutes when only code changes.
#
# Build:
#   docker build -t youruser/infinitetalk:single-1.0 \
#     --build-arg MODELS_IMAGE=youruser/infinitetalk-models:single-1.0 .
#   docker build -t youruser/infinitetalk:multi-1.0 \
#     --build-arg MODELS_IMAGE=youruser/infinitetalk-models:multi-1.0 .

# Set this to the tag you pushed via models.Dockerfile.
# Override locally with: docker build --build-arg MODELS_IMAGE=...
ARG MODELS_IMAGE=youruser/infinitetalk-models:single-1.0
FROM ${MODELS_IMAGE} AS models

FROM wlsdml1114/engui_genai-base_blackwell:1.1 AS runtime

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]" && \
    pip install runpod websocket-client librosa

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

# Models from the prebuilt models image. Single layer copy.
COPY --from=models /models /ComfyUI/models

COPY . .
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
