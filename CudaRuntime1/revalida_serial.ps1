# Revalidacao: serial CORRIGIDO no MESMO Beta da GPU (R0=3.5). Confirma equivalencia no L real.
$ErrorActionPreference = 'Stop'
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64;$env:PATH"
$ser = "C:\Users\User\source\repos\CudaRuntime1\CudaRuntime1"
$val = "C:\Users\User\source\repos\CudaRuntime1-gpu\benchmarks\curvas\validacao"
Set-Location $ser
$files = 'epidemicsprevalence.dat','epidemicsincidence.dat','Infectiousprevalence.dat','Infectiousincidence.dat'
# cidade -> L, Beta(GPU R0=3.5), ataque GPU de referencia
$cfg = @(
  @{c='BRA'; L=1604; b=0.0995; gpu=0.7675},
  @{c='MAN'; L=1343; b=0.0995; gpu=0.7591}
)
foreach ($x in $cfg) {
  (Get-Content kernel.cu) -replace 'cities\(\w+\);', "cities($($x.c));" | Set-Content kernel.cu
  $bs = $x.b.ToString([System.Globalization.CultureInfo]::InvariantCulture)
  (Get-Content define.h) -replace 'const int L = \d+;', "const int L = $($x.L);" `
    -replace 'const int MAXSIM = \d+;', 'const int MAXSIM = 50;' `
    -replace 'const double Beta = [0-9.]+;', "const double Beta = $bs;" | Set-Content define.h
  nvcc kernel.cu -O3 -o serial_sim.exe -arch=sm_89 --diag-suppress 20091 -Xcompiler "/O2 /wd4716" 2>$null | Out-Null
  $t = Get-Date
  .\serial_sim.exe *> $null
  $sec = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
  $dst = Join-Path $val "$($x.c)\serial"
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  foreach ($f in $files) { Copy-Item (Join-Path $ser $f) (Join-Path $dst $f) -Force }
  $last = ((Get-Content "$ser\epidemicsprevalence.dat") | Select-Object -Last 1) -split "`t"
  $atk = [math]::Round(1 - [double]$last[1], 4)
  "[SERIAL $($x.c) corrigido] beta=$bs L=$($x.L) MAXSIM=50 tempo=${sec}s ataque=$atk (GPU=$($x.gpu) dif=$([math]::Round($atk-$x.gpu,4)))" |
    Tee-Object -Append (Join-Path $val 'revalida_log.txt')
}
"=== REVALIDA BRA+MAN CONCLUIDO $(Get-Date -Format 'HH:mm:ss') ===" | Tee-Object -Append (Join-Path $val 'revalida_log.txt')
