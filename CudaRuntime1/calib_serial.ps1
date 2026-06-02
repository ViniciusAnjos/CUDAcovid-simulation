# Sweep de calibracao do Beta serial (full-sim) p/ casar o ataque-alvo da GPU.
# Uso: edita define.h (Beta), compila com otimizacao, roda, le ataque. Loga cada ponto.
param(
    [double[]]$Betas = @(0.105, 0.120, 0.135),
    [int]$Maxsim = 10,
    [string]$Tag = "BRA"
)
$ErrorActionPreference = 'Stop'
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64;$env:PATH"
$ser = "C:\Users\User\source\repos\CudaRuntime1\CudaRuntime1"
Set-Location $ser
$log = Join-Path $ser "calib_serial_$Tag.log"
"=== Calibracao serial $Tag MAXSIM=$Maxsim inicio=$(Get-Date -Format 'HH:mm:ss') ===" | Out-File $log -Encoding ASCII

# fixa MAXSIM de busca
(Get-Content define.h) -replace 'const int MAXSIM = \d+;', "const int MAXSIM = $Maxsim;" | Set-Content define.h

foreach ($b in $Betas) {
    $bs = $b.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    (Get-Content define.h) -replace 'const double Beta = [0-9.]+;', "const double Beta = $bs;" | Set-Content define.h
    nvcc kernel.cu -O3 -o serial_sim.exe -arch=sm_89 --diag-suppress 20091 -Xcompiler "/O2 /wd4716" 2>$null | Out-Null
    $t = Get-Date
    .\serial_sim.exe *> $null
    $sec = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    $last = ((Get-Content epidemicsprevalence.dat) | Select-Object -Last 1) -split "`t"
    $atk = [math]::Round(1 - [double]$last[1], 4)
    # fracao de sims que decolaram (ataque>0.3)
    $pf = Get-ChildItem -Filter 'prevalence_*.dat'
    $tk = 0; foreach ($f in $pf) { $l = (Get-Content $f.FullName | Select-Object -Last 1) -split "`t"; if ((1 - [double]$l[1]) -gt 0.3) { $tk++ } }
    "Beta=$bs  ataque=$atk  decolaram=$tk/$($pf.Count)  tempo=${sec}s" | Tee-Object -Append $log
}
"=== fim $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append $log -Encoding ASCII
