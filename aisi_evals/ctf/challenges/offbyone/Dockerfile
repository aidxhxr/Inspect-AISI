FROM python:3.12-slim
COPY keygen.py /srv/service/keygen.py
COPY expected.sha256 /srv/service/expected.sha256
RUN chmod 444 /srv/service/keygen.py /srv/service/expected.sha256
WORKDIR /srv/service
