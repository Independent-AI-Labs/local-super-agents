# Delete the VM
$VMName = "AgentsVM"
$VHDXPath = "C:\ProgramData\Microsoft\Windows\Virtual Hard Disks\AgentsVM.vhdx"

# Check if VM exists
$VM = Get-VM -Name $VMName -ErrorAction SilentlyContinue
if ($VM) {
    # If the VM is running, shut it down gracefully first
    if ($VM.State -eq "Running") {
        Write-Output "Shutting down VM '$VMName'..."
        Stop-VM -Name $VMName -Force
    }

    # Remove the VM
    Remove-VM -Name $VMName -Force
    Write-Output "VM '$VMName' has been removed."
} else {
    Write-Output "VM '$VMName' does not exist."
}

# Delete the VHDX file if it exists
if (Test-Path $VHDXPath) {
    Remove-Item -Path $VHDXPath -Force
    Write-Output "VHDX file '$VHDXPath' has been deleted."
} else {
    Write-Output "VHDX file '$VHDXPath' does not exist."
}

Write-Output "Cleanup complete."

# Remove port forwarding rules
# Remove-NetNatStaticMapping -NatName "AgentsNATNetwork" -Confirm:$true

# Remove NAT network
Remove-NetNat -Name "AgentsNATNetwork" -Confirm:$false

# Remove assigned IP address
$InterfaceIndex = (Get-NetAdapter -Name "vEthernet (AgentsNAT)").IfIndex
Remove-NetIPAddress -InterfaceIndex $InterfaceIndex -Confirm:$false

# Remove the virtual switch
Remove-VMSwitch -Name "AgentsNAT" -Confirm:$false
