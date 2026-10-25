# Деплой

## systemd (VPS)

    sudo mkdir -p /opt/DanyAI
    sudo rsync -a --exclude runs --exclude data/raw ./ /opt/DanyAI/
    cd /opt/DanyAI && python3 -m venv .venv && .venv/bin/pip install .
    mkdir -p runs/sft && cp best_sft.npz runs/sft/best.npz
    sudo cp deploy/danyai.service /etc/systemd/system/
    sudo systemctl enable --now danyai

Снаружи: cloudflared tunnel или nginx с TLS, ключ задаётся флагом --api-key.

## Docker

    docker build -f deploy/Dockerfile -t danyai .
    docker run -p 8080:8080 -e DANYAI_API_KEY=secret danyai

## Termux (Android)

    bash deploy/termux.sh
    termux-wake-lock

Скрипт ждёт чекпоинт в models/model.npz и токенизатор в models/tokenizer.json.
Без systemd автозапуск делается через termux-boot: положить `bash ~/DanyAI/deploy/termux.sh` в ~/.termux/boot/start-danyai.

Проверка:

    curl -s localhost:8080/health
    curl -s -X POST localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"Once upon a time","max_tokens":32}'
