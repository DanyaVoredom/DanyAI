param(
    [string]$Serial = "adb-RFCT80JEWRF-ARaxpN._adb-tls-connect._tcp",
    [string]$Adb = "C:\Users\Dmode\AppData\Local\Microsoft\WinGet\Packages\Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe\platform-tools\adb.exe",
    [string]$Script = "~/.termux/boot/danyai"
)

$ErrorActionPreference = "Stop"
& $Adb start-server | Out-Null

& $Adb -s $Serial shell "input keyevent 224" | Out-Null
Start-Sleep -Milliseconds 800
& $Adb -s $Serial shell "monkey -p com.termux -c android.intent.category.LAUNCHER 1" | Out-Null
Start-Sleep -Seconds 3

$cmd = "bash $Script"
$typed = $cmd -replace " ", "%s"
$typed = $typed -replace "~", "\~"
& $Adb -s $Serial shell "input text '$typed'" | Out-Null
Start-Sleep -Milliseconds 500
& $Adb -s $Serial shell "input keyevent 66" | Out-Null
Start-Sleep -Seconds 6

for ($i = 0; $i -lt 10; $i++) {
    $ok = (Test-NetConnection -ComputerName "192.168.0.112" -Port 8022 -WarningAction SilentlyContinue).TcpTestSucceeded
    if ($ok) {
        Write-Output "sshd поднялся, попытка $i"
        exit 0
    }
    Start-Sleep -Seconds 3
}
Write-Output "sshd не поднялся, проверь экран терминала"
exit 1
