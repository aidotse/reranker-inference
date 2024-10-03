# Reranker Inference Service

Reranker inference service intended for use with the Digital Assistant. Simply hosts a
reranker model using HuggingFace transformers and exposes a prediction endpoint.

## Build

```sh
make build
```

## Running

To run in the project use

```sh
make run
```

When running in production, use

```sh
docker volume create hf_cache  # If not exists
docker run -it -p 5000:5000 -v hf_cache:/app/hf_cache --gpus all -e API_KEY=<token> ghcr.io/aidotse/reranker-inference:latest
```

## Push

```sh
make push
```
