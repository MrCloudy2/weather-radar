"""Idempotently insert the /radar/ location into the nextcloud_nginx config.

Backs up the file, inserts the block just before `location /nextcloud {` inside
the 443 server block. Safe to run repeatedly.
"""
import pathlib
import shutil
import sys

CONF = pathlib.Path("/home/cloudy/nextcloud/nginx.conf")
ANCHOR = "    location /nextcloud {"
BLOCK = """    location /radar/ {
        proxy_pass http://172.18.0.1:8099/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
    location = /radar { return 301 /radar/; }

"""

text = CONF.read_text()
if "location /radar/" in text:
    print("already present — no change")
    sys.exit(0)
if ANCHOR not in text:
    print("ANCHOR NOT FOUND — aborting, no change")
    sys.exit(1)
shutil.copy(str(CONF), str(CONF) + ".bak-radar")
CONF.write_text(text.replace(ANCHOR, BLOCK + ANCHOR, 1))
print("inserted /radar/ location (backup: nginx.conf.bak-radar)")
