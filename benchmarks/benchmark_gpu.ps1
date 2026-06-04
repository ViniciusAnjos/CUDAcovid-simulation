# =====================================================================
# Benchmark portatil da GPU (so GPU) - roda as cidades nesta GPU e compara
# com a referencia da RTX 4070 SUPER. Detecta arch e cl.exe automaticamente.
#
# Uso (no diretorio benchmarks/ do repo clonado):
#   powershell -ExecutionPolicy Bypass -File .\benchmark_gpu.ps1
#   powershell -ExecutionPolicy Bypass -File .\benchmark_gpu.ps1 -Maxsim 5 -Cidades ROC,MAN,BRA,SP
# =====================================================================
param(
    [int]$Maxsim = 5,
    [string[]]$Cidades = @("ROC", "MAN", "BRA", "SP")
)
$ErrorActionPreference = "Stop"
# permite -Cidades ROC,MAN (que via -File chega como uma string unica)
if ($Cidades.Count -eq 1 -and $Cidades[0] -match ',') { $Cidades = $Cidades[0] -split '\s*,\s*' }
$src = Join-Path $PSScriptRoot "..\CudaRuntime1"
if (-not (Test-Path (Join-Path $src "covid.cu"))) { Write-Error "Nao achei CudaRuntime1\covid.cu (rode de benchmarks\)"; exit 1 }

# --- 1. localizar cl.exe (Visual Studio C++) ---
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vs) {
            $cldir = Get-ChildItem "$vs\VC\Tools\MSVC\*\bin\HostX64\x64" -Directory -ErrorAction SilentlyContinue | Select-Object -Last 1
            if ($cldir) { $env:PATH = "$($cldir.FullName);$env:PATH" }
        }
    }
}
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) { Write-Error "cl.exe nao encontrado. Instale o 'Desktop development with C++' no Visual Studio."; exit 1 }
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    $nv = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\nvcc.exe" -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($nv) { $env:PATH = "$(Split-Path $nv.FullName);$env:PATH" } else { Write-Error "nvcc nao encontrado. Instale o CUDA Toolkit."; exit 1 }
}

# --- 2. detectar GPU e arch ---
$smi = "C:\Windows\System32\nvidia-smi.exe"
$gpuName = (& $smi --query-gpu=name --format=csv,noheader).Trim()
$cc = (& $smi --query-gpu=compute_cap --format=csv,noheader 2>$null)
if ($cc) { $cc = $cc.Trim(); $sm = "sm_" + ($cc -replace '\.', '') } else { $sm = "sm_61"; $cc = "?" }
$vram = (& $smi --query-gpu=memory.total --format=csv,noheader).Trim()
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " GPU: $gpuName" -ForegroundColor Cyan
Write-Host " compute capability: $cc  ->  -arch=$sm   |  VRAM: $vram" -ForegroundColor Cyan
Write-Host " MAXSIM=$Maxsim  |  cidades: $($Cidades -join ', ')" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# --- 3. configs e referencia (RTX 4070 SUPER, s/sim e ataque) ---
$cfg = @{
    ROC = @{ nome = "Rocinha";   L = 264;  b = 0.0049 }
    MAN = @{ nome = "Manaus";    L = 1343; b = 0.0995 }
    BRA = @{ nome = "Brasilia";  L = 1604; b = 0.0995 }
    SP  = @{ nome = "Sao Paulo"; L = 3355; b = 0.0243 }
}
$ref4070 = @{ ROC = @{t = 2.57; atk = 0.587 }; MAN = @{t = 2.71; atk = 0.759 }; BRA = @{t = 2.63; atk = 0.7675 }; SP = @{t = 12.67; atk = 0.726 } }

Set-Location $src
$res = @()
foreach ($c in $Cidades) {
    if (-not $cfg.ContainsKey($c)) { Write-Host "cidade $c desconhecida, pulando"; continue }
    $L = $cfg[$c].L; $b = $cfg[$c].b
    $bs = $b.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    (Get-Content covid.cu) -replace 'int city = \w+;', "int city = $c;" | Set-Content covid.cu
    (Get-Content define.h) -replace 'const int L = \d+;', "const int L = $L;" `
        -replace 'const int MAXSIM = \d+;', "const int MAXSIM = $Maxsim;" `
        -replace 'const double Beta = [0-9.]+;', "const double Beta = $bs;" | Set-Content define.h
    Write-Host "`n[$c] $($cfg[$c].nome)  L=$L  compilando ($sm)..." -ForegroundColor Yellow
    Remove-Item covid_sim.exe -ErrorAction SilentlyContinue   # evita rodar exe antigo se a compilacao falhar
    $nvout = nvcc covid.cu -allow-unsupported-compiler -o covid_sim.exe "-arch=$sm" --diag-suppress 20091 --diag-suppress 177 2>&1
    if (-not (Test-Path covid_sim.exe)) {
        Write-Host "  FALHA na compilacao de ${c}:" -ForegroundColor Red
        $nvout | Select-Object -Last 8 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
        continue
    }
    $t0 = Get-Date
    .\covid_sim.exe *> "run_$c.txt"
    $ec = $LASTEXITCODE
    $sec = ((Get-Date) - $t0).TotalSeconds
    if ($ec -ne 0) {
        Write-Host "  [$c] FALHOU (exit=$ec) - provavel TDR (kernel > 2s). Aumente o TdrDelay (ver README)." -ForegroundColor Red
        $res += [PSCustomObject]@{ Cidade = $c; tsim = "TDR"; ataque = "-"; vs4070 = "-" }
        continue
    }
    $last = ((Get-Content epidemicsprevalence.dat) | Select-Object -Last 1) -split "`t"
    $atk = [math]::Round(1 - [double]$last[1], 4)
    $tsim = [math]::Round($sec / $Maxsim, 2)
    $ratio = [math]::Round($tsim / $ref4070[$c].t, 1)
    Write-Host ("  [$c] tempo/sim={0}s  ataque={1}  (4070S: {2}s/{3})  -> {4}x mais lento" -f $tsim, $atk, $ref4070[$c].t, $ref4070[$c].atk, $ratio) -ForegroundColor Green
    $res += [PSCustomObject]@{ Cidade = $c; tsim = $tsim; ataque = $atk; vs4070 = "${ratio}x" }
}

Write-Host "`n===================== RESUMO ($gpuName) =====================" -ForegroundColor Cyan
$res | Format-Table -AutoSize
Write-Host "Referencia 4070S (s/sim): ROC 2.57 | MAN 2.71 | BRA 2.63 | SP 12.67"
Write-Host "O ataque DEVE bater com a 4070S (RNG deterministico) -> valida independencia de hardware."
$res | ConvertTo-Json | Out-File (Join-Path $PSScriptRoot "resultado_$($gpuName -replace '[^\w]','_').json")
Write-Host "Salvo em benchmarks\resultado_*.json - mande esse arquivo para juntar ao benchmark." -ForegroundColor Cyan
