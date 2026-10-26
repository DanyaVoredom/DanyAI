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

    bash deploy/termux_services.sh
    termux-wake-lock

Скрипт регистрирует runit-сервисы danyai-sshd, danyai-server, danyai-webchat и danyai-train:
они перезапускаются после падения процесса, логи пишутся в ~/DanyAI/logs/<сервис>/current,
плюс создаётся ~/.termux/boot/danyai для автозапуска после перезагрузки (нужно приложение Termux:Boot).

    sv status danyai-server
    tail -f ~/DanyAI/logs/danyai-train/current

Никаких pkg install и тяжёлых бенчмарков на телефоне: при обучении занято 6-8 ГБ из 11,
Android выгружает Termux вместе с sshd. Пакеты ставятся только на ПК, на телефоне допускаются
только ssh-команды чтения и перезапуск сервисов.

## Android не убивает Termux

Android 12+ убивает дочерние процессы Termux (phantom process killer), а doze выгружает приложение
вместе с sshd. Лечится только через adb, изнутри Termux эти настройки недоступны:

    deploy\immunity.ps1 -Serial 192.168.0.112:5555

Перед запуском включить на телефоне Параметры разработчика - Беспроводная отладка, подключиться
по коду из этого же экрана. Скрипт снимает Termux с doze, ставит standby bucket active, разрешает
фоновую работу и wake lock, выключает phantom process killer и поднимает лимит фантомных процессов.
После перезагрузки настройки сохраняются.

Вручную то же самое:

    adb shell cmd deviceidle whitelist +com.termux
    adb shell am set-standby-bucket com.termux active
    adb shell settings put global settings_enable_monitor_phantom_procs false
    adb shell device_config put activity_manager max_phantom_processes 2147483647

Ещё в интерфейсе: Termux - Батарея - Без ограничений, и снять галочку "Удалять разрешения,
если приложение не используется".

sshd теперь тоже под runit: если процесс умрёт, он поднимется сам. При первом запуске скрипта
убить старый sshd, иначе новый не сможет занять порт:

    pkill sshd; bash deploy/termux_services.sh

Скрипт ждёт чекпоинт в models/model.npz и токенизатор в models/tokenizer.json.
Без systemd автозапуск делается через termux-boot: положить `bash ~/DanyAI/deploy/termux.sh` в ~/.termux/boot/start-danyai.

Проверка:

    curl -s localhost:8080/health
    curl -s -X POST localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"Once upon a time","max_tokens":32}'
