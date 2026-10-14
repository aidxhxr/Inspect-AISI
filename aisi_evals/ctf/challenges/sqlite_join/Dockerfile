FROM python:3.12-slim
ARG FLAG
COPY make_db.py /tmp/make_db.py
RUN python /tmp/make_db.py "$FLAG" && rm /tmp/make_db.py
WORKDIR /srv/data
