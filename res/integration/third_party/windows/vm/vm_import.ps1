# ---------------------------------------------------------------
# PS file to import a Hyper-V VM with detailed debugging
# ---------------------------------------------------------------

function Execute-Step {
    param (
        [string]$StepName,
        [scriptblock]$Command
    )
    
    Write-Host ""
    Write-Host ">>> Starting: $StepName" -ForegroundColor Cyan
    try {
        & $Command
        Write-Host "Completed: $StepName" -ForegroundColor Green
        # Short pause to ensure command finishes
        Start-Sleep -Seconds 1 
    } catch {
        Write-Host "ERROR: Step failed: $StepName" -ForegroundColor Red
        Write-Host ("Details: " + $_.Exception.Message) -ForegroundColor Yellow
        exit 1
    }
	
	
}

# 0. Check if NetNat 'AgentsNATNetwork' exists, if so, run cleanup script
Execute-Step "Checking for existing NAT network 'AgentsNATNetwork'" {
    $existingNat = Get-NetNat -Name "AgentsNATNetwork" -ErrorAction SilentlyContinue
    if ($existingNat) {
        Write-Host "Existing NAT network found. Running cleanup script..."
        $cleanupScript = "$env:install_path\agents\res\integration\third_party\windows\vm\vm_cleanup.ps1"
        if (Test-Path $cleanupScript) {
            powershell -ExecutionPolicy Bypass -File $cleanupScript
            Write-Host "Cleanup script executed successfully."
        } else {
            throw "Cleanup script not found at $cleanupScript"
        }
    } else {
        Write-Host "No existing NAT network found, proceeding with setup."
    }
}

# 1. Create an internal virtual switch
Execute-Step "Creating internal switch 'AgentsNAT'" {
    New-VMSwitch -SwitchName "AgentsNAT" -SwitchType Internal
}

# 2. Get the interface index of the new vEthernet adapter
Execute-Step "Getting Interface Index for 'AgentsNAT'" {
    $InterfaceIndex = (Get-NetAdapter -Name "vEthernet (AgentsNAT)").IfIndex
    Write-Host "Interface Index found: $InterfaceIndex"
}

$InterfaceIndex = (Get-NetAdapter -Name "vEthernet (AgentsNAT)").IfIndex

# 3. Assign IP address to the interface
Execute-Step "Assigning IP 172.72.72.1/24 to the 'AgentsNAT' adapter" {
    New-NetIPAddress -IPAddress 172.72.72.1 -PrefixLength 24 -InterfaceIndex $InterfaceIndex
}

# 4. Create a NAT network
Execute-Step "Creating NAT network 'AgentsNATNetwork'" {
    New-NetNat -Name "AgentsNATNetwork" -InternalIPInterfaceAddressPrefix "172.72.72.0/24"
}

# 5. Add port forwarding rules
# Execute-Step "Adding port forwarding (8888 -> 172.72.72.2:8888)" {
#     Add-NetNatStaticMapping `
#         -NatName "AgentsNATNetwork" `
#         -Protocol TCP `
#         -ExternalIPAddress "127.0.0.1" `
#         -ExternalPort 8888 `
#         -InternalIPAddress "172.72.72.2" `
#         -InternalPort 8888
# }

# Execute-Step "Adding port forwarding (8000 -> 172.72.72.2:8000)" {
#     Add-NetNatStaticMapping `
#         -NatName "AgentsNATNetwork" `
#         -Protocol TCP `
#         -ExternalIPAddress "127.0.0.1" `
#         -ExternalPort 8000 `
#         -InternalIPAddress "172.72.72.2" `
#         -InternalPort 8000
# }

# 6. Path to the folder containing the exported VM configuration
# Define the directory path
$VMDirectory = "$env:install_path\agents\res\integration\third_party\windows\vm\AgentsVM\Virtual Machines"

# Get the latest .vmcx file in the directory
$VMXFile = Get-ChildItem -Path $VMDirectory -Filter "*.vmcx" | Select-Object -First 1

# Construct the full path to the VM configuration file
if ($VMXFile) {
    $VMPath = Join-Path -Path $VMDirectory -ChildPath $VMXFile.Name
    Write-Output "Found VM configuration file: $VMPath"
} else {
    Write-Error "No .vmcx file found in $VMDirectory"
    exit 1
}

Execute-Step "Verifying VM configuration file exists at $VMPath" {
    if (-not (Test-Path $VMPath)) {
        throw "VM configuration file not found at $VMPath"
    }
}

# 7. Import the VM and copy files
Execute-Step "Importing VM from $VMPath" {
    $VM = Import-VM -Path $VMPath -Copy
}

# Start the VM
Start-VM -Name "AgentsVM"

# Set the VM to always start automatically with the host
Set-VM -Name "AgentsVM" -AutomaticStartAction Start

Write-Host "`nAgentsVM installed successfully."
