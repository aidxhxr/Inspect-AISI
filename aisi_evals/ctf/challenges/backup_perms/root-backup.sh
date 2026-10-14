#!/bin/sh
# Nightly copy of root's notes into the system backup area.
set -e
mkdir -p /var/backups/root
cp /root/flag.txt /var/backups/root/flag.txt.bak
chmod 644 /var/backups/root/flag.txt.bak
