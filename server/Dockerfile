# FROM python:3.12-slim
FROM tensorflow/tensorflow:latest-gpu
# Instalar dependências
RUN apt update
RUN apt install -y wget build-essential libbz2-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev
RUN apt install -y software-properties-common
RUN apt install -y build-essential
RUN apt install -y libssl-dev
RUN apt install -y zlib1g-dev
RUN apt install -y libncurses5-dev
RUN apt install -y libgdbm-dev
RUN apt install -y libnss3-dev
RUN apt install -y libsqlite3-dev
RUN apt install -y libreadline-dev
RUN apt install -y libffi-dev
RUN apt install -y curl
RUN apt install -y libbz2-dev
RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tar.xz
RUN tar -xf Python-3.12.0.tar.xz
WORKDIR /Python-3.12.0
RUN ./configure --enable-optimizations
RUN make -j 8
RUN make altinstall
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Instalar TensorFlow com suporte a GPU
RUN python3.12 -m pip install tensorflow


RUN mkdir app
RUN mkdir app/strategies
RUN pip install --upgrade pip
RUN pip install numpy flwr scikit-learn
CMD ["python3.12", "-u", "/app/strategies_manager.py"]