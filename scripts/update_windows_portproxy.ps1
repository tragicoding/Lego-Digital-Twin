param(
    [Parameter(Mandatory = $true)]
    [string]$WslIp,

    [int[]]$Ports = @(3000, 3001, 8000)
)

$ErrorActionPreference = "Stop"

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    Write-Host "Requesting Windows administrator approval for portproxy update..."
    $portsCsv = ($Ports -join ",")
    $quotedScriptPath = '"' + $PSCommandPath + '"'
    $argString = "-NoProfile -ExecutionPolicy Bypass -File $quotedScriptPath -WslIp $WslIp -Ports $portsCsv"

    try {
        $process = Start-Process -FilePath "powershell.exe" -Verb RunAs -Wait -PassThru -ArgumentList $argString
    }
    catch {
        Write-Error "Administrator approval was not granted."
        exit 1
    }

    if ($null -eq $process) {
        Write-Error "Failed to launch elevated PowerShell."
        exit 1
    }

    exit $process.ExitCode
}

Write-Host "Updating Windows portproxy rules for WSL IP $WslIp"

foreach ($port in $Ports) {
    & netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 | Out-Null
    & netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$WslIp | Out-Null

    $ruleName = "WSL2 $port"
    $existingRule = & netsh advfirewall firewall show rule name=$ruleName
    if ($existingRule -match "No rules match") {
        & netsh advfirewall firewall add rule name=$ruleName dir=in action=allow protocol=TCP localport=$port profile=any | Out-Null
    }

    Write-Host "  Port $port -> $WslIp`:$port"
}

Write-Host "Windows portproxy update complete."
