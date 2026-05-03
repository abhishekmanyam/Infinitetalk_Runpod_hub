# Builds a thin image whose only job is to hold the model weights.
# Use it as a stage source in the runtime Dockerfile so code-only rebuilds
# never re-download the ~30 GB of weights.
#
# Build:
#   docker build -t youruser/infinitetalk-models:single-1.0 \
#     --build-arg VARIANT=single -f models.Dockerfile .
#   docker build -t youruser/infinitetalk-models:multi-1.0 \
#     --build-arg VARIANT=multi  -f models.Dockerfile .
#   docker push youruser/infinitetalk-models:single-1.0
#   docker push youruser/infinitetalk-models:multi-1.0

FROM python:3.10-slim AS downloader

ARG VARIANT=single
ENV VARIANT=${VARIANT}
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV MODELS_DIR=/models

RUN pip install --no-cache-dir "huggingface_hub[hf_transfer]"

WORKDIR /work
COPY download_models.py .

RUN python download_models.py

# Final stage: scratch keeps the image to just the weights, nothing else.
FROM scratch
COPY --from=downloader /models /models
