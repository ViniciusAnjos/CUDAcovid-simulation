# Aumenta o TdrDelay do WDDM para 60s (precisa rodar elevado). Reboot necessario depois.
$k = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
New-ItemProperty -Path $k -Name TdrDelay    -PropertyType DWord -Value 60 -Force | Out-Null
New-ItemProperty -Path $k -Name TdrDdiDelay -PropertyType DWord -Value 60 -Force | Out-Null
$p = Get-ItemProperty -Path $k
$marker = 'C:\Users\User\source\repos\CudaRuntime1-gpu\benchmarks\tdr_set_marker.txt'
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  TdrDelay=$($p.TdrDelay)  TdrDdiDelay=$($p.TdrDdiDelay)  (reboot necessario)" |
    Out-File $marker -Encoding ASCII
