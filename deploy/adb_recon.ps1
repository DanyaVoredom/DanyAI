#!/usr/bin/env pwsh
# Собирает всё, что нужно для решения о руте и разблокировке загрузчика.
param(
    [string]$Serial = "",
    [string]$Out = ""
)

$ErrorActionPreference = "Continue"

$platform = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools"
$adb = (Get-Command adb -ErrorAction SilentlyContinue).Source
if (-not $adb -and (Test-Path "$platform\adb.exe")) { $adb = "$platform\adb.exe" }
$fastboot = (Get-Command fastboot -ErrorAction SilentlyContinue).Source
if (-not $fastboot -and (Test-Path "$platform\fastboot.exe")) { $fastboot = "$platform\fastboot.exe" }
if (-not $adb) { throw "adb не найден" }

if (-not $Out) { $Out = Join-Path $env:TEMP ("adb_recon_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss")) }
$script:log = New-Object System.Collections.Generic.List[string]

function W($line) { $script:log.Add($line); Write-Output $line }
function A($args_) { & $adb @adbArgs @args_ 2>&1 }

$script:adbArgs = @()
if ($Serial) { $script:adbArgs = @("-s", $Serial) }

& $adb start-server | Out-Null
$devices = (& $adb devices) -join "`n"
W "== adb devices"
W $devices

$props = @(
    "ro.product.brand", "ro.product.manufacturer", "ro.product.model", "ro.product.device",
    "ro.product.name", "ro.product.board", "ro.board.platform", "ro.hardware", "ro.boot.hardware",
    "ro.build.version.release", "ro.build.version.sdk", "ro.build.version.security_patch",
    "ro.build.fingerprint", "ro.build.type", "ro.build.tags", "ro.build.display.id",
    "ro.secure", "ro.debuggable", "ro.adb.secure", "service.adb.root",
    "ro.boot.verifiedbootstate", "ro.boot.flash.locked", "ro.boot.veritymode",
    "ro.boot.vbmeta.device_state", "ro.boot.slot_suffix", "ro.crypto.state",
    "ro.oem_unlock_supported", "sys.oem_unlock_allowed", "ro.boot.hwc", "ro.boot.project_name",
    "ro.miui.has_real_blur", "ro.twrp.version", "ro.allow.mock.location"
)

W "`n== свойства"
foreach ($p in $props) {
    $v = (A @("shell", "getprop $p") | Out-String).Trim()
    if ($v) { W ("{0} = {1}" -f $p, $v) }
}

W "`n== признаки платформы"
W ((A @("shell", "getprop | grep -iE 'mt[0-9]{4}|mediatek|mtk|qcom|qualcomm|exynos|unisoc|sprd|trusty|tee|avb'") | Out-String).Trim())

W "`n== cpu"
W ((A @("shell", "cat /proc/cpuinfo | grep -iE 'hardware|processor|model name' | head -12") | Out-String).Trim())

W "`n== разделы"
W ((A @("shell", "ls -l /dev/block/by-name 2>/dev/null | head -40") | Out-String).Trim())
W ((A @("shell", "cat /proc/partitions | head -30") | Out-String).Trim())

W "`n== защита загрузчика"
W ((A @("shell", "ls -l /dev/block/by-name/vbmeta* /dev/block/by-name/boot /dev/block/by-name/recovery 2>/dev/null") | Out-String).Trim())
W ((A @("shell", "dumpsys android.hardware.boot.IBootControl/default 2>/dev/null | head -20") | Out-String).Trim())

W "`n== сom selinux и рут-маркеры"
W ((A @("shell", "getenforce; id; which su; ls /system/xbin/su /system/bin/su /sbin/su 2>/dev/null") | Out-String).Trim())

W "`n== пользователи и права"
W ((A @("shell", "pm list users") | Out-String).Trim())

W "`n== fastboot"
if ($fastboot) {
    W ((& $fastboot devices 2>&1 | Out-String).Trim())
} else {
    W "fastboot не найден в PATH и в platform-tools"
}

$script:log | Set-Content -Path $Out -Encoding utf8
Write-Output "`nотчёт: $Out"
