# Define variables
$vmName = "AgentsVM"
$exportPath = Join-Path -Path (Get-Location) -ChildPath $vmName
$archivePath = Join-Path -Path (Get-Location) -ChildPath "$vmName.7z"
$sevenZipPath = Join-Path -Path ${env:ProgramFiles} -ChildPath "7-Zip\7z.exe"

# Export the VM
Export-VM -Name $vmName -Path $exportPath

# Check if 7-Zip is installed
if (-Not (Test-Path -Path $sevenZipPath)) {
    Write-Error "7-Zip executable not found at $sevenZipPath. Please ensure 7-Zip is installed."
    exit 1
}

# Compress the exported VM into a split 7z archive
& $sevenZipPath a -t7z "$archivePath" "$exportPath\*" -v2000m

# Verify if the archive was created successfully
if ($?) {
    # Remove the exported VM files
    Remove-Item -Path $exportPath -Recurse -Force
    Write-Output "Exported VM files have been archived and deleted successfully."
} else {
    Write-Error "An error occurred during the archiving process. Exported VM files have not been deleted."
}