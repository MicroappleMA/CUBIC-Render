$sourcePath = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition

Write-Output "Start Decompressing Lib Under Folder: $sourcePath"

# Get all .lib.zip files in the directory and its sub directories
$filesToDecompress = Get-ChildItem -Path $sourcePath -Recurse -Filter *.lib.zip 

# Decompress each file individually
foreach ($zipfile in $filesToDecompress) {
    $destinationPath = $zipfile.FullName.TrimEnd(".zip")

    # if the .lib file already exists, delete it
    if (Test-Path $destinationPath) {
        Write-Warning "Lib $destinationPath already exists and will be deleted."
        Remove-Item $destinationPath -Force
    }

    # Decompress the .lib.zip file
    Expand-Archive -Path $zipfile.FullName -DestinationPath (Split-Path $destinationPath -Parent) -Force

    # Delete the original .lib.zip file
    Remove-Item $zipfile.FullName -Force

    Write-Output "Decompressed Lib: $destinationPath"
}

Write-Output "Stop Decompressing"