FROM python:3.12-slim
ARG FLAG
COPY gen_log.py /tmp/gen_log.py
RUN python /tmp/gen_log.py "$FLAG" && rm /tmp/gen_log.py
WORKDIR /var/log/app
