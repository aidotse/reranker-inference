# Reranker Inference Service

Reranker inference service intended for use with the Digital Assistant.

## Build

```sh
docker build -t ghcr.io/aidotse/reranker-inference:latest .
```

## Running

```sh
docker run -it -p 5000:5000 -v hf_cache:/app/hf_cache --gpus all reranker-inference:latest
```
