param(
    [string]$AdapterName = "",
    [switch]$RestoreAutoNegotiation,
    [switch]$SkipDns
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-TargetAdapter {
    param([string]$Name)

    if ($Name) {
        return Get-NetAdapter -Name $Name
    }

    $adapter = Get-NetAdapter |
        Where-Object { $_.InterfaceDescription -match "I226-V" } |
        Sort-Object @{ Expression = { $_.Status -eq "Up" }; Descending = $true }, Name |
        Select-Object -First 1

    if (-not $adapter) {
        throw "No Intel I226-V adapter was found."
    }

    return $adapter
}

if (-not (Test-IsAdmin)) {
    throw "This script must be run from an elevated PowerShell window."
}

$adapter = Get-TargetAdapter -Name $AdapterName
$speedTarget = if ($RestoreAutoNegotiation) { "自动协商" } else { "1.0 Gbps 全双工" }

Write-Host "[network-fix] adapter: $($adapter.Name) / $($adapter.InterfaceDescription)"
Write-Host "[network-fix] target speed/duplex: $speedTarget"

$speedProperty = Get-NetAdapterAdvancedProperty -Name $adapter.Name -RegistryKeyword "*SpeedDuplex"
if ($speedProperty.DisplayValue -ne $speedTarget) {
    Set-NetAdapterAdvancedProperty -Name $adapter.Name -DisplayName $speedProperty.DisplayName -DisplayValue $speedTarget
}

if (-not $SkipDns) {
    Write-Host "[network-fix] setting public DNS servers: 1.1.1.1, 8.8.8.8"
    Set-DnsClientServerAddress -InterfaceAlias $adapter.Name -ServerAddresses @("1.1.1.1", "8.8.8.8")
}

Write-Host "[network-fix] flushing DNS cache"
Clear-DnsClientCache
ipconfig /flushdns | Out-Null

Start-Sleep -Seconds 8

$adapter = Get-NetAdapter -Name $adapter.Name
$speedProperty = Get-NetAdapterAdvancedProperty -Name $adapter.Name -RegistryKeyword "*SpeedDuplex"

Write-Host "[network-fix] adapter status after change"
$adapter | Select-Object Name, Status, LinkSpeed, InterfaceDescription | Format-Table -AutoSize

Write-Host "[network-fix] speed/duplex property after change"
$speedProperty | Select-Object DisplayName, DisplayValue | Format-Table -AutoSize

Write-Host "[network-fix] DNS servers after change"
Get-DnsClientServerAddress -InterfaceAlias $adapter.Name -AddressFamily IPv4 |
    Select-Object InterfaceAlias, ServerAddresses |
    Format-Table -AutoSize

Write-Host "[network-fix] verifying connectivity"
Test-NetConnection 8.8.8.8 -InformationLevel Detailed |
    Select-Object ComputerName, InterfaceAlias, SourceAddress, PingSucceeded, PingReplyDetails |
    Format-List

Test-NetConnection openai.com -Port 443 -InformationLevel Detailed |
    Select-Object ComputerName, RemoteAddress, RemotePort, InterfaceAlias, SourceAddress, TcpTestSucceeded |
    Format-List

Write-Host "[network-fix] done"
