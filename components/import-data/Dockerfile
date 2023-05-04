FROM python:3.7-alpine
LABEL authors="Jonathan Bailey"

RUN apk add  --no-cache --update alpine-sdk cargo snappy-dev

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY ./src /pipelines/component/src