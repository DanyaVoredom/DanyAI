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

## Рут на телефоне

Проверено на SM-F936B (Galaxy Z Fold4, Snapdragon 8+ Gen 1, Android 16, One UI 8):

    ro.build.type=user, ro.debuggable=0    adb root недоступен
    verifiedbootstate green, flash.locked  загрузчик залочен, vbmeta locked, verity enforcing
    oem_lock service                       любой вызов: SecurityException, нужно
                                           android.permission.MANAGE_CARRIER_OEM_UNLOCK_STATE

Samsung не реализует fastboot, разблокировка только через download mode после переключателя
OEM unlocking, а в One UI 8 его убрали из интерфейса. Обойти из adb shell нельзя: единственный
вызывающий oem_lock с нужным правом - системный Settings. BROM-эксплойты (mtkclient) только для
MediaTek, здесь Qualcomm с подписанным firehose, публичного загрузчика для F936B нет.
Итог: рута и разблокировки на этом устройстве нет, и для задачи они не нужны.

Что пробовали по части подмены приложений с правами (все попытки упираются в подпись и SELinux):

    свой apk с MANAGE_USER_OEM_UNLOCK_STATE    ставится, право не выдаётся,
    и MANAGE_CARRIER_OEM_UNLOCK_STATE          ServiceManager.getService("oem_lock") возвращает null -
                                               SELinux не даёт домену приложения даже увидеть службу
    переподпись системного apk своим ключом    INSTALL_FAILED_UPDATE_INCOMPATIBLE,
                                               подписи не совпадают
    android:sharedUserId=android.uid.system    INSTALL_FAILED_SHARED_USER_INCOMPATIBLE,
                                               нет подписи, совпадающей с shared user
    adb shell pm grant <право>                 SecurityException: not a changeable permission type
    adb shell service call oem_lock ...        SecurityException: нужно
                                               MANAGE_CARRIER_OEM_UNLOCK_STATE

Обойти это можно только заменив apk в /system или подменив подпись платформы, а для этого уже нужен
root. Круг замкнут.

Метод с подменой abl.elf.lz4 из One UI 7 в BL от One UI 8 (droidwin) для этого телефона не работает
из-за anti-rollback. Разрядность прошивки это 5-й символ с конца PDA:

    F936BXXSAHYI1  One UI 7 (Android 15)   bit A
    F936BXXUAHYD9  One UI 7 (Android 15)   bit A
    F936BXXUAIYIF  One UI 8 (Android 16)   bit A
    F936BXXSBIZD1  One UI 8, 2026-04       bit B
    F936BXXSCIZH3  стоит сейчас, 2026-08   bit C

Все прошивки One UI 7 для F936B имеют bit A, а на телефоне bit C. Понижение bit блокирует
загрузчик (SW REV CHECK FAIL), поэтому abl от One UI 7 туда не встанет. Штатного пути к
переключателю OEM на этом устройстве нет ни через adb, ни через Odin.

## llama.cpp и litellm на телефоне

Инференс через llama.cpp поверх GGUF, а litellm работает шлюзом с одним OpenAI-совместимым адресом
для всех моделей DanyAI.

    bash deploy/build_llama_termux.sh        # cmake + сборка llama-server, один раз
    bash deploy/termux_services.sh           # регистрирует danyai-llama (8081) и danyai-litellm (4000)

Модель качается отдельно (1.31 ГБ, Q4_K_M):

    curl -L -C - -o ~/models_gguf/qwen35-2b-q4km.gguf \
      https://huggingface.co/noTrabajo/Huihui-Qwen3.5-2B-abliterated-Q4_K_M-GGUF/resolve/main/huihui-qwen3.5-2b-abliterated-q4_k_m.gguf

LoRA из safetensors в GGUF собирается на ПК (нужен torch и клон llama.cpp):

    python tools/export_lora_gguf.py --llama /path/to/llama.cpp --adapter runs/lora/lora \
      --out ~/models_gguf/danyai-lora.gguf

Если файл ~/models_gguf/danyai-lora.gguf существует, danyai-llama подхватит его через --lora.
Сервисы danyai-llama и danyai-litellm сами уходят в sleep, пока нет бинарника или модели,
так что runit не крутит их в цикле перезапуска.

    curl -s localhost:8081/v1/models
    curl -s -H "Authorization: Bearer sk-danyai-local" localhost:4000/v1/models

Модели в litellm: danyai-qwen (llama.cpp, 8081) и danyai-nano (свой numpy-сервер, 8080).
Ключ и список моделей в configs/litellm.yaml.

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

sshd поднимается сам только если живой runsvdir: когда Android убивает приложение, вместе с ним
умирает вся ветка runit, и перезапускать некому. Поэтому лечится не сервисом, а тем, чтобы приложение
не убивали. Штатный сервис sshd из termux-services включается один раз:

    sv-enable sshd

Скрипт ждёт чекпоинт в models/model.npz и токенизатор в models/tokenizer.json.
Без systemd автозапуск делается через termux-boot: положить `bash ~/DanyAI/deploy/termux.sh` в ~/.termux/boot/start-danyai.

Проверка:

    curl -s localhost:8080/health
    curl -s -X POST localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"Once upon a time","max_tokens":32}'
