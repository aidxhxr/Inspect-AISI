FROM python:3.12-slim
ARG FLAG
COPY populate.py /tmp/populate.py
RUN python /tmp/populate.py "$FLAG" && rm /tmp/populate.py
WORKDIR /opt/app
