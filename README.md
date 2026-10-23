# DanyAI

Локальная линейка языковых моделей, обученных с нуля. Реализация на numpy, без torch. CPU и GPU, один и тот же код.

## Линейка

| Модель | Слои | Головы | d_model | Параметры при vocab 4096 |
|---|---|---|---|---|
| DanyAI-Nano | 4 | 4 | 128 | 1.35M |
| DanyAI-Mini | 6 | 8 | 256 | 5.85M |
| DanyAI-Base | 8 | 6 | 384 | 15.86M |

Параметры зависят от словаря: почти половина весов в эмбеддинге wte, он же используется как выходная голова.

## Состав

- danyai/tensor.py - autograd на numpy, проверен численно
- danyai/model.py - GPT-декодер: MHA, MLP, pre-LN, tied embeddings
- danyai/tokenizer.py - BPE с инкрементальным подсчётом пар и кэшем
- danyai/prepare.py - корпус txt в train.bin/val.bin и tokenizer.json
- danyai/train.py - AdamW, warmup, cosine, gradient clipping, best.npz по val
- danyai/sample.py, danyai/chat.py - генерация и REPL
- danyai/server.py - HTTP API, включая /v1/completions
- danyai/bench.py - замер скорости шага
- tests/gradcheck.py - численная проверка градиентов, зелёный
- tests/smoke.py - сквозной прогон на синтетике

## Пайплайн

    python tools/fetch_corpus.py
    python -m danyai.prepare --input data/raw --out data --vocab 4096 --val 0.02
    python -m danyai.train --config configs/nano_run.json --data data --out runs/nano
    python -m danyai.sample --ckpt runs/nano/best.npz --tok data/tokenizer.json --prompt "The house" --tokens 200
    python -m danyai.chat --ckpt runs/nano/best.npz --tok data/tokenizer.json
    python -m danyai.server --ckpt runs/nano/best.npz --tok data/tokenizer.json

## Проверки

    python tests/gradcheck.py
    python tests/smoke.py

## Железо

i7-2630QM, 14 GB RAM, без GPU. Замеры на этой машине, блок 128, батч 32: около 4.3 с на шаг, порядка 950 токенов в секунду. Прогон Nano на 3000 шагов занимает примерно 3.5 часа.

Для полноценного обучения нужен GPU. colab/train_danyai.ipynb содержит готовый ноутбук: клонирование, скачивание корпуса, подготовка, обучение Mini на 20000 шагов, сэмплинг. T4 даёт ускорение в десятки раз, код не меняется.