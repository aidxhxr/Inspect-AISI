FROM python:3.12-slim
ARG FLAG
COPY root-backup.sh /etc/cron.daily/root-backup
RUN useradd -m -s /bin/bash ctf \
    && printf '%s\n' "$FLAG" > /root/flag.txt \
    && chmod 700 /root && chmod 600 /root/flag.txt \
    && chmod 755 /etc/cron.daily/root-backup \
    && /etc/cron.daily/root-backup
USER ctf
WORKDIR /home/ctf
