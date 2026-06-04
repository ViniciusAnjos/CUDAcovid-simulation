# Roda as 4 cidades (GPU) MAXSIM=50 salvando os 50 prevalence_N.dat por cidade
# para analise de variabilidade (bandas de confianca + histograma de ataque/bimodalidade).
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64;$env:PATH"
$gpu = "C:\Users\User\source\repos\CudaRuntime1-gpu\CudaRuntime1"
$out = "C:\Users\User\source\repos\CudaRuntime1-gpu\benchmarks\curvas\variabilidade"
Set-Location $gpu
$log = Join-Path $out "_log.txt"
New-Item -ItemType Directory -Force -Path $out | Out-Null
"=== variabilidade inicio $(Get-Date -Format 'HH:mm:ss') ===" | Out-File $log -Encoding ASCII

$cfg = @(
    @{c = "ROC"; L = 264;  b = "0.0049" },
    @{c = "MAN"; L = 1343; b = "0.0995" },
    @{c = "BRA"; L = 1604; b = "0.0995" },
    @{c = "SP";  L = 3355; b = "0.0243" }
)
foreach ($x in $cfg) {
    (Get-Content covid.cu) -replace 'int city = \w+;', "int city = $($x.c);" | Set-Content covid.cu
    (Get-Content define.h) -replace 'const int L = \d+;', "const int L = $($x.L);" `
        -replace 'const int MAXSIM = \d+;', 'const int MAXSIM = 50;' `
        -replace 'const double Beta = [0-9.]+;', "const double Beta = $($x.b);" | Set-Content define.h
    Remove-Item covid_sim.exe -ErrorAction SilentlyContinue
    nvcc covid.cu -o covid_sim.exe -arch=sm_89 --diag-suppress 20091 --diag-suppress 177 2>$null | Out-Null
    if (-not (Test-Path covid_sim.exe)) { "[$($x.c)] FALHA compilacao" | Tee-Object -Append $log; continue }
    $t = Get-Date
    .\covid_sim.exe *> $null
    $sec = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    $dst = Join-Path $out $x.c
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Get-ChildItem "$gpu\prevalence_*.dat" | Copy-Item -Destination $dst -Force
    $nsaved = (Get-ChildItem "$dst\prevalence_*.dat").Count
    "[$($x.c)] L=$($x.L) beta=$($x.b) MAXSIM=50 tempo=${sec}s  brutos_salvos=$nsaved" | Tee-Object -Append $log
}
"=== variabilidade DONE $(Get-Date -Format 'HH:mm:ss') ===" | Tee-Object -Append $log
