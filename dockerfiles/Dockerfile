FROM nvidia/cuda:11.1.1-base-ubuntu20.04
RUN apt update && apt install -y python3 python3-pip sudo

COPY . /workspace

RUN /bin/bash -c "cd workspace \
    && pip install -r requirements.txt"
