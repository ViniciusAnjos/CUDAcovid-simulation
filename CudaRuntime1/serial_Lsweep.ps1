# Sweep de L no serial (config SP, MAXSIM=3) para curva tempo x L / speedup x L.
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64;$env:PATH"
$ser = "C:\Users\User\source\repos\CudaRuntime1\CudaRuntime1"; Set-Location $ser
$out = "C:\Users\User\source\repos\CudaRuntime1-gpu\benchmarks\curvas\graficos\hardware\Lsweep_serial.txt"
"L`tt_sim_s`ttotal_s" | Out-File $out -Encoding ASCII
(Get-Content kernel.cu) -replace 'cities\(\w+\);','cities(SP);' | Set-Content kernel.cu
foreach ($L in 200,400,800,1600,3200) {
    (Get-Content define.h) -replace 'const int L = \d+;', "const int L = $L;" `
        -replace 'const int MAXSIM = \d+;', 'const int MAXSIM = 3;' `
        -replace 'const double Beta = [0-9.]+;', 'const double Beta = 0.0243;' | Set-Content define.h
    nvcc kernel.cu -O3 -o serial_sim.exe -arch=sm_89 --diag-suppress 20091 -Xcompiler "/O2 /wd4716" 2>$null | Out-Null
    $t = Get-Date; .\serial_sim.exe *> $null; $sec = ((Get-Date) - $t).TotalSeconds
    "$L`t$([math]::Round($sec/3,2))`t$([math]::Round($sec,1))" | Tee-Object -Append $out
}
"=== serial Lsweep DONE $(Get-Date -Format HH:mm:ss) ===" | Out-File -Append $out -Encoding ASCII
