FROM python:3.9

RUN mkdir /eshiritori-ml
COPY . /eshiritori-ml/
WORKDIR /eshiritori-ml/

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN python3 -m spacy download en_core_web_lg
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116