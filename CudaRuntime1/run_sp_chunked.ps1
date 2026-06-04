# Driver resiliente a TDR para Sao Paulo (L=3355).
# Roda em chunks curtos; retry em chunk que travar (TDR -> exit !=0); media dos chunks no fim.
$ErrorActionPreference = 'Stop'
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64;$env:PATH"
$gpu = "C:\Users\User\source\repos\CudaRuntime1-gpu\CudaRuntime1"
$cur = "C:\Users\User\source\repos\CudaRuntime1-gpu\benchmarks\curvas"
Set-Location $gpu

$chunkSize  = 5
$numChunks  = 6          # 6 x 5 = 30 sims
$maxRetries = 4
$files = 'epidemicsprevalence.dat','epidemicsincidence.dat','Infectiousprevalence.dat','Infectiousincidence.dat'
$chunkDir = Join-Path $cur 'SP_chunks'
New-Item -ItemType Directory -Force -Path $chunkDir | Out-Null
Get-ChildItem $chunkDir -Filter *.dat -ErrorAction SilentlyContinue | Remove-Item -Force

$tStart = Get-Date
$ok = @()
for ($c = 1; $c -le $numChunks; $c++) {
    $seedBase = ($c - 1) * $chunkSize
    $done = $false
    for ($try = 1; $try -le $maxRetries -and -not $done; $try++) {
        $t = Get-Date
        & .\covid_sim.exe $chunkSize $seedBase *> "chunk_run.txt"
        $ec = $LASTEXITCODE
        $sec = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
        $prevOk = (Test-Path epidemicsprevalence.dat) -and ((Get-Item epidemicsprevalence.dat).Length -gt 0)
        if ($ec -eq 0 -and $prevOk) {
            foreach ($f in $files) { Copy-Item $f (Join-Path $chunkDir ("c{0}_{1}" -f $c, $f)) -Force }
            $last = ((Get-Content epidemicsprevalence.dat) | Select-Object -Last 1) -split "`t"
            $atk = [math]::Round(1 - [double]$last[1], 4)
            "[CHUNK $c/$numChunks] OK try=$try seedBase=$seedBase tempo=${sec}s ataque=$atk" |
                Tee-Object -Append (Join-Path $cur 'sp_chunked_log.txt')
            $ok += $c
            $done = $true
        } else {
            "[CHUNK $c/$numChunks] FALHOU try=$try exit=$ec tempo=${sec}s (TDR?) -> retry" |
                Tee-Object -Append (Join-Path $cur 'sp_chunked_log.txt')
            Start-Sleep -Seconds 3   # deixa o driver assentar apos reset
        }
    }
}

# --- Media element-wise dos chunks que deram certo ---
function Average-DatFiles($pattern, $outPath) {
    $cfiles = Get-ChildItem $chunkDir -Filter $pattern | Sort-Object Name
    if ($cfiles.Count -eq 0) { return $false }
    $all = $cfiles | ForEach-Object { Get-Content $_.FullName }
    $nLines = ($all[0]).Count
    # cada $all[i] e um array de linhas; assume mesmo numero de linhas
    $lineSets = @($cfiles | ForEach-Object { ,(Get-Content $_.FullName) })
    $nRows = $lineSets[0].Count
    $header = $lineSets[0][0]
    $out = New-Object System.Collections.Generic.List[string]
    if ($header -match '[A-Za-z]') { $out.Add($header); $startRow = 1 } else { $startRow = 0 }
    for ($r = $startRow; $r -lt $nRows; $r++) {
        $cols0 = $lineSets[0][$r] -split "`t"
        $nc = $cols0.Count
        $acc = New-Object double[] $nc
        foreach ($set in $lineSets) {
            $cols = $set[$r] -split "`t"
            for ($k = 0; $k -lt $nc; $k++) { $acc[$k] += [double]$cols[$k] }
        }
        $vals = for ($k = 0; $k -lt $nc; $k++) {
            if ($k -eq 0) { [int][math]::Round($acc[0] / $lineSets.Count) }   # day
            else { ($acc[$k] / $lineSets.Count) }
        }
        $out.Add(($vals -join "`t"))
    }
    Set-Content -Path $outPath -Value $out -Encoding ASCII
    return $true
}

New-Item -ItemType Directory -Force -Path (Join-Path $cur 'SP') | Out-Null
foreach ($f in $files) {
    $pat = "c*_$f"
    Average-DatFiles $pat (Join-Path $cur "SP\$f") | Out-Null
}

$totalSec = [math]::Round(((Get-Date) - $tStart).TotalSeconds, 1)
$nOk = $ok.Count
$simsOk = $nOk * $chunkSize
$last = ((Get-Content (Join-Path $cur 'SP\epidemicsprevalence.dat')) | Select-Object -Last 1) -split "`t"
$rows = Get-Content (Join-Path $cur 'SP\epidemicsprevalence.dat') | Select-Object -Skip 1 |
        ForEach-Object { $a = $_ -split "`t"; [PSCustomObject]@{d=[int]$a[0]; Inf=[double]$a[5]; H=[double]$a[6]; ICU=[double]$a[7]} }
$pInf = $rows | Sort-Object Inf -Descending | Select-Object -First 1
"=== SP CHUNKED CONCLUIDO === chunks_ok=$nOk/$numChunks sims=$simsOk tempo_total=${totalSec}s ataque=$([math]::Round(1-[double]$last[1],4)) picoInf=$([math]::Round($pInf.Inf,4))(d$($pInf.d)) Rec=$([double]$last[8]) Mortes=$([double]$last[9])" |
    Tee-Object -Append (Join-Path $cur 'sp_chunked_log.txt')
