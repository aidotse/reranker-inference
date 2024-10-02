FROM nvcr.io/nvidia/ai-workbench/python-cuda122:1.0.3

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY src /app/src

CMD ["python3", "/app/src/app.py"]
