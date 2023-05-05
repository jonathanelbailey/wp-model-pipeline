FROM python:3.9
LABEL authors="Jonathan Bailey"

COPY requirements.txt .
RUN pip3 install -r requirements.txt --no-cache-dir

WORKDIR /pipelines/components/src

COPY src/. .

CMD ["python3"]