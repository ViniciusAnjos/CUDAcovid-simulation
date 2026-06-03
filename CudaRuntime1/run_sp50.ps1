# SP serial MAXSIM=50 overnight, com estatisticas de tempo por simulacao (benchmark).
# Tempo por sim derivado do mtime de prevalence_N.dat (robusto: sobrevive a falha do wrapper).
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64;$env:PATH"
$ser = "C:\Users\User\source\repos\CudaRuntime1\CudaRuntime1"
$val = "C:\Users\User\source\repos\CudaRuntime1-gpu\benchmarks\curvas\validacao"
Set-Location $ser
$prog = Join-Path $ser "sp50_progress.txt"
$start = Get-Date
"START $($start.ToString('o'))" | Out-File $prog -Encoding ASCII

.\serial_sim.exe *> (Join-Path $ser "sp50_run.txt")
$ec = $LASTEXITCODE
$end = Get-Date
$total = [math]::Round(($end - $start).TotalSeconds, 1)

# tempo por simulacao a partir do mtime de prevalence_N.dat
$pf = Get-ChildItem "$ser\prevalence_*.dat" | Sort-Object { [int]($_.BaseName -replace 'prevalence_','') }
$prev = $start; $persim = @()
foreach ($f in $pf) { $dt = [math]::Round(($f.LastWriteTime - $prev).TotalSeconds,1); $persim += $dt; $prev = $f.LastWriteTime }
$mean = if($persim.Count){ [math]::Round(($persim | Measure-Object -Average).Average,1) } else { 0 }

# copia curvas finais (sobrescreve o MAXSIM=10)
New-Item -ItemType Directory -Force -Path "$val\SP\serial" | Out-Null
foreach ($x in 'epidemicsprevalence.dat','epidemicsincidence.dat','Infectiousprevalence.dat','Infectiousincidence.dat') { Copy-Item "$ser\$x" "$val\SP\serial\$x" -Force }
$ls = ((Get-Content "$ser\epidemicsprevalence.dat") | Select-Object -Last 1) -split "`t"
$atk = [math]::Round(1 - [double]$ls[1], 4)

$stats = Join-Path $val "sp50_benchmark.txt"
"=== SP SERIAL MAXSIM=50 (overnight) ===" | Out-File $stats -Encoding ASCII
"exit=$ec  inicio=$($start.ToString('HH:mm:ss'))  fim=$($end.ToString('HH:mm:ss'))" | Out-File $stats -Append -Encoding ASCII
"tempo_total=${total}s ($([math]::Round($total/60,1)) min)  sims=$($pf.Count)  tempo_medio/sim=${mean}s" | Out-File $stats -Append -Encoding ASCII
"ataque=$atk (GPU=0.726)" | Out-File $stats -Append -Encoding ASCII
"tempo por sim (s): " + ($persim -join ', ') | Out-File $stats -Append -Encoding ASCII
Get-Content $stats
"DONE total=${total}s atk=$atk" | Out-File $prog -Append -Encoding ASCII
