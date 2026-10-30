#!/usr/bin/env bash
DB=/home/lipton/DanyAI/colab-profile/Default/Cookies
cp "$DB" /tmp/c.db 2>/dev/null || cp "$DB" /tmp/c.db
python3 - <<'EOF'
import sqlite3, datetime
c = sqlite3.connect("/tmp/c.db")
rows = c.execute("select host_key,name,expires_utc,length(encrypted_value) from cookies where name in ('SID','HSID','SSID','SAPISID','__Secure-1PSID','__Secure-3PSID')").fetchall()
for h,n,e,l in rows:
    if e:
        dt = datetime.datetime(1601,1,1) + datetime.timedelta(microseconds=e)
    else:
        dt = None
    print(h, n, dt, l)
print("total", c.execute("select count(*) from cookies").fetchone()[0])
EOF
ls -la /home/lipton/DanyAI/colab-profile/Default/Cookies
