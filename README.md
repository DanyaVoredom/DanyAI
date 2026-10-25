# DanyAI

Локальная линейка языковых моделей, обученных с нуля. Реализация на numpy, без torch. Считает на CPU, при наличии CUDA работает через cupy.

## Линейка

| Модель | Слои | Головы | d_model | Параметры при vocab 8192 |
|---|---|---|---|---|
| DanyAI-Nano | 4 | 4 | 128 | ~2.0M |
| DanyAI-Mini | 6 | 8 | 256 | ~7.5M |
| DanyAI-Base | 8 | 6 | 384 | ~17M |

Параметры зависят от словаря: почти половина весов в эмбеддинге wte, он же используется как выходная голова.

## Состав

- danyai/tensor.py - autograd на numpy, проверен численно
- danyai/backend.py - выбор вычислительного бэкенда: numpy (CPU) или cupy (CUDA)
- danyai/model.py - GPT-декодер: MHA/GQA, MLP/SwiGLU, pre-LN/RMSNorm, RoPE, tied embeddings
- danyai/tokenizer.py - BPE с инкрементальным подсчётом пар и кэшем
- danyai/prepare.py - корпус txt в train.bin/val.bin и tokenizer.json
- danyai/train.py - AdamW, warmup, cosine, gradient clipping, best.npz по val
- danyai/infer.py - движок генерации на numpy: KV-cache, сэмплинг, perplexity, int8
- danyai/sample.py, danyai/chat.py - генерация и REPL
- danyai/sft.py - дообучение на парах инструкция/ответ с маскировкой лосса
- danyai/eval.py - perplexity по val.bin
- danyai/server.py - OpenAI-совместимый HTTP API с SSE-стримингом
- danyai/webchat.py - веб-чат на stdlib
- danyai/bench.py - замер скорости шага
- tools/export_onnx.py - экспорт в ONNX
- tools/quantize.py - int8-квантование чекпоинта
- tests/gradcheck.py - численная проверка градиентов
- tests/smoke.py - сквозной прогон на синтетике

## Пайплайн

    python tools/fetch_corpus.py
    python -m danyai.prepare --input data/raw --out data --vocab 8192 --val 0.02
    python -m danyai.train --config configs/modern.json --data data --out runs/modern
    python -m danyai.eval --ckpt runs/modern/best.npz --data data
    python -m danyai.sample --ckpt runs/modern/best.npz --tok data/tokenizer.json --prompt "The house" --tokens 200

Сэмплинг: `--temp`, `--top-k`, `--top-p`, `--repetition`, `--presence`, `--frequency`, `--seed`.

## SFT и чат

    python tools/make_sft.py --src data/raw/tinystories/TinyStories-valid.txt --out data/sft.jsonl --limit 8000
    python -m danyai.sft --ckpt runs/modern/best.npz --data data/sft.jsonl --tok data/tokenizer.json --out runs/sft --steps 600
    python -m danyai.chat --ckpt runs/sft/best.npz --tok data/tokenizer.json

Теги диалога: `<|user|>`, `<|assistant|>`, `<|end|>`.

## Сервер и веб-чат

    python -m danyai.server --ckpt runs/sft/best.npz --tok data/tokenizer.json --host 0.0.0.0 --port 8080 --api-key SECRET
    python -m danyai.webchat --ckpt runs/sft/best.npz --tok data/tokenizer.json --port 8090

Эндпоинты: `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/generate`. Стриминг через `"stream": true` (SSE).

## GPU

По умолчанию считается на CPU. Для CUDA:

    pip install cupy-cuda12x
    DANYAI_GPU=1 python -m danyai.train --config configs/modern.json --data data --out runs/modern

Флаг `DANYAI_GPU=1` переключает autograd на cupy, чекпоинты по-прежнему сохраняются в numpy и читаются на CPU.

## Colab из консоли

    pip install playwright
    python tools/colab_cli.py --headed --keep-open

Первый запуск в headed-режиме нужен один раз для входа в Google. Профиль сохраняется в %LOCALAPPDATA%\DanyAI\colab-profile. Скрипт собирает данные в gist, открывает ноутбук, включает GPU-бэкенд, запускает обучение и SFT, стримит логи и забирает чекпоинты в runs/. Флаги: `--steps`, `--batch`, `--block`, `--timeout`, `--connect-wait`, `--out`, `--no-artifacts`.

## Экспорт

    python tools/export_onnx.py --ckpt runs/modern/best.npz --out runs/onnx/model.onnx
    python tools/quantize.py --ckpt runs/modern/best.npz --out runs/int8/model_int8.npz

## Результаты

Modern-2M на корпусе vocab 8192 (6 486 016 train-токенов), обучение на T4 через cupy, 2000 шагов:

| Чекпоинт | val loss | perplexity |
|---|---|---|
| danyai-modern-2m | 1.6441 | 5.18 |
| danyai-modern-2m-sft | 1.6576 | - |

Артефакты релиза: `danyai-modern-2m.npz`, `danyai-modern-2m-sft.npz`, `danyai-modern-2m-sft.onnx`, `danyai-modern-2m-sft-int8.npz`, `tokenizer.json`, `meta.json`, `config.json`. ONNX совпадает с numpy-движком (max_abs 2e-5, argmax 100%), int8 - 0.46 и 98.4% argmax.

## Хостинг

Модель маленькая, хватает 1-2 vCPU и 1-2 GB RAM. Готовые файлы в deploy/: systemd-юнит и Dockerfile. Снаружи закрывать TLS и авторизацию через reverse-proxy или cloudflared.

## Проверки

    python tests/gradcheck.py
    python tests/smoke.py

## Железо

i7-2630QM, 14 GB RAM. Замеры на этой машине, блок 128, батч 32, OPENBLAS_NUM_THREADS=8: 2.9 с на шаг, порядка 1400 токенов в секунду. Видеокарта GeForce GTX 460M (Fermi) для обучения непригодна.
