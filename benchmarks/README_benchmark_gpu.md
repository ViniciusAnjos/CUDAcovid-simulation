# Benchmark portátil da GPU (rodar em outra placa)

Mede o tempo da **simulação na GPU** desta máquina nas 4 cidades e compara com a referência da
**RTX 4070 SUPER**. Útil para o gráfico *speedup × modelo de GPU* da monografia.

## Pré-requisitos
1. **GPU NVIDIA** (placa integrada/AMD não roda CUDA).
2. **CUDA Toolkit** instalado (`nvcc`).
3. **Visual Studio** com *Desktop development with C++* (fornece o `cl.exe`).
4. **Git**.

## Como rodar (Windows / PowerShell)
```powershell
git clone https://github.com/ViniciusAnjos/CUDAcovid-simulation.git
cd CUDAcovid-simulation
git checkout fix/state-machine-timing
cd benchmarks
powershell -ExecutionPolicy Bypass -File .\benchmark_gpu.ps1
```
O script **detecta sozinho** a arquitetura da GPU (`-arch=sm_XX`) e o `cl.exe`.

### Opções
```powershell
# menos simulações (mais rápido) e só algumas cidades:
powershell -ExecutionPolicy Bypass -File .\benchmark_gpu.ps1 -Maxsim 3 -Cidades ROC,MAN,BRA
```
- `-Maxsim N` (padrão 5): nº de simulações por cidade. Menos = mais rápido.
- `-Cidades ROC,MAN,BRA,SP` (padrão todas).

## Saída
Imprime uma tabela com **tempo/sim**, **ataque** e quantas vezes mais lento que a 4070 SUPER, e
salva um `resultado_<GPU>.json`. **Mande esse JSON** para juntar ao benchmark.

> O **ataque deve bater** com a 4070 SUPER (RNG determinístico) — isso valida que o resultado
> independe do hardware. As referências (s/sim, MAXSIM=50): ROC 2,57 · MAN 2,71 · BRA 2,63 · SP 12,67.

## ⚠️ GPUs mais lentas (ex.: GTX 1050 Ti) — atenção ao TDR
Em placas lentas, um kernel da **São Paulo** (L=3355) pode passar de 2 s e o Windows **reseta o
driver** (TDR). O script detecta (a cidade aparece como `TDR`) e segue. Para evitar:
- Rode sem a SP: `-Cidades ROC,MAN,BRA`; **ou**
- Aumente o timeout do TDR (precisa admin + **reboot**):
  ```
  reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f
  ```
- VRAM: SP usa ~780 MB; em placas de 4 GB cabe (L máximo ≈ 7900).
