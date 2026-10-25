# HF / LoRA бэкенд

`danyai.hf_server` поднимает OpenAI-совместимый сервер поверх transformers
(обычные и мультимодальные модели, LoRA-адаптеры peft).

    python -m danyai.hf_server --model huihui-ai/Huihui-Qwen3.5-2B-abliterated \
        --adapter runs/lora --host 0.0.0.0 --port 8080 --api-key secret

Эндпоинты: `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/generate`.
Поддержка `stream: true` (SSE) и `tools` (tool calling в формате OpenAI).

Запрос с инструментами:

    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -H 'X-Api-Key: secret' -d '{
      "messages": [{"role": "user", "content": "Какая погода в Москве?"}],
      "tools": [{"type": "function", "function": {"name": "get_weather", "description": "weather by city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    }'

Ответ при вызове инструмента содержит `finish_reason: tool_calls` и `message.tool_calls`
(блоки `<tool_call>` разбираются автоматически).

Обучение адаптера:

    python tools/finetune_lora.py --model huihui-ai/Huihui-Qwen3.5-2B-abliterated \
        --data data/sft.jsonl,data/tools.jsonl --out runs/lora \
        --max-len 768 --batch 4 --grad-accum 4 --epochs 1

Адаптер не содержит базовых весов, поэтому для инференса нужны и модель, и каталог адаптера.
