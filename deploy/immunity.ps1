#!/usr/bin/env pwsh
# Ослабляет вмешательство Android в Termux: phantom process killer, doze, standby bucket.
# Запуск с ПК при включённой отладке по USB или Wireless debugging.
param(
    [string]$Serial = "192.168.0.112:5555",
    [string]$Adb = "adb"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command $Adb -ErrorAction SilentlyContinue)) {
    $cand = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
    if (Test-Path $cand) { $Adb = $cand } else { throw "adb не найден: поставь platform-tools" }
}

& $Adb start-server | Out-Null
$state = (& $Adb devices) -join "`n"
if ($state -notmatch [regex]::Escape($Serial) + "\s+device") {
    & $Adb connect $Serial | Out-Null
    Start-Sleep -Seconds 2
}
$state = (& $Adb devices) -join "`n"
if ($state -notmatch [regex]::Escape($Serial) + "\s+device") {
    throw "нет adb-доступа к $Serial. Включи Отладку по USB или Беспроводную отладку и подтверди ключ на телефоне"
}

$A = @("-s", $Serial, "shell")

function Run($desc, $cmd) {
    $out = & $Adb @A $cmd 2>&1
    Write-Output ("{0}: {1}" -f $desc, (($out | Out-String).Trim()))
}

Run "android" "getprop ro.build.version.release"
Run "sdk" "getprop ro.build.version.sdk"
Run "doze whitelist termux" "cmd deviceidle whitelist +com.termux"
Run "standby bucket" "am set-standby-bucket com.termux active"
Run "appops RUN_IN_BACKGROUND" "cmd appops set com.termux RUN_IN_BACKGROUND allow"
Run "appops RUN_ANY_IN_BACKGROUND" "cmd appops set com.termux RUN_ANY_IN_BACKGROUND allow"
Run "appops WAKE_LOCK" "cmd appops set com.termux WAKE_LOCK allow"
Run "phantom monitor off" "settings put global settings_enable_monitor_phantom_procs false"
Run "max_phantom_processes" "device_config put activity_manager max_phantom_processes 2147483647"

$sdk = [int]((& $Adb @A "getprop ro.build.version.sdk" | Out-String).Trim())
if ($sdk -ge 31) {
    Run "phantom native boot" "device_config put activity_manager_native_boot max_phantom_processes 2147483647"
}

Run "doze whitelist dump" "cmd deviceidle whitelist"
Write-Output "готово, часть настроек окончательно применится после перезагрузки телефона"
