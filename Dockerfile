FROM python:3.10

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y cmake libboost-all-dev libprotobuf-dev libopenblas-dev g++ git autoconf automake libtool curl make g++ unzip

RUN git clone --branch v1.0.0 https://github.com/NeuralNetworkVerification/Marabou.git marabou

COPY setup.py download_protobuf.sh ./
RUN python setup.py

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD [ "python", "./main.py" ]
# CMD [ "bash" ]