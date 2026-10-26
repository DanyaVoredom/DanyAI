# Modal

Обучение на GPU через Modal. Артефакты не выкладываются на файлохостинги: они пишутся в Modal volume и забираются командой `modal volume get`.

## Установка клиента

С этого ПК api.modal.com не отвечает на TLS-хендшейк (python, curl, schannel таймаут; Chrome и телефон проходят), поэтому клиент ставится на телефон:

    pkg install -y rust          # watchfiles собирается из исходников
    pip install modal
    python -m modal setup

Токен создаётся в браузере на ПК. Callback-порт телефона пробрасывается:

    ssh -N -L 8686:127.0.0.1:8686 homevps

## Обучение 20M

    cd ~/DanyAI
    python -m modal run tools/modal_train.py --out big2

Датасет data-v2 скачивается один раз в volume danyai-data, чекпоинты лежат в danyai-runs.

Забрать веса:

    python -m modal volume get danyai-runs big2/best.npz /tmp/big2/best.npz
    python -m modal volume get danyai-runs big2/sft/best.npz /tmp/big2/sft-best.npz

## LoRA на Qwen3.5-2B abliterated

    python -m modal run tools/modal_lora.py --out lora --epochs 1.0 --max-len 512 --batch 2

Готовый архив адаптера: `danyai-runs/lora.zip`.

    python -m modal volume get danyai-runs lora.zip /tmp/lora.zip
    python -m tools.check_adapter --zip /tmp/lora.zip

## Проверка

    python -m modal run tools/modal_train.py --gpu-job eval --out big2

## Обновление кода в образе

Образ собирается из ветки dev на GitHub, локальные правки не подхватываются. После изменений: push в dev, затем любой запуск пересобирает слой.
