IMAGE_NAME = ghcr.io/aidotse/reranker-inference
GIT_SHA = $(shell git rev-parse --short HEAD)

.PHONY: build
build:
	docker build -t ${IMAGE_NAME}:${GIT_SHA} -t ${IMAGE_NAME}:latest .

.PHONY: push
push:
	docker push ${IMAGE_NAME} --all-tags

.PHONY: run
run:
	docker run -it -p 5000:5000 -v hf_cache:/app/hf_cache --gpus all ${IMAGE_NAME}:latest
