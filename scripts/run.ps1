# Runner standar: menjalankan script python dan menulis log otomatis ke logs\.
# Pemakaian:  .\run.ps1 nama_script.py [argumen...]
# Log:        logs\<nama_script>_<timestamp>.log
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ScriptArgs
)
$ErrorActionPreference = "Stop"
$base = "E:\Download\Jurnal"
$py = "E:\pyvenv_geo\Scripts\python.exe"
$scriptPath = Join-Path $base "scripts\$Script"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    Write-Error "Script tidak ditemukan: $scriptPath"
    exit 1
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logName = [System.IO.Path]::GetFileNameWithoutExtension($Script) + "_" + $stamp + ".log"
$logPath = Join-Path $base "logs\$logName"
Write-Output "Menjalankan: $Script"
Write-Output "Log: $logPath"
# stderr python (traceback/warning) tidak boleh mematikan runner di PS 5.1:
# konversi tiap baris ke string sebelum Tee.
$ErrorActionPreference = "Continue"
& $py $scriptPath @ScriptArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $logPath
$code = $LASTEXITCODE
Write-Output "Exit code: $code"
exit $code