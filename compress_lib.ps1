$sourcePath = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition

Write-Output "Start Compressing Lib Under Folder: $sourcePath"

# Get all .lib files in the directory and its sub directories
$filesToCompress = Get-ChildItem -Path $sourcePath -Recurse -Filter *.lib 

# Compress each file individually
foreach ($file in $filesToCompress) {
    $destinationPath = $file.FullName + ".zip"

    # If the .lib.zip file already exists, delete it
    if (Test-Path $destinationPath) {
        Remove-Item $destinationPath -Force
    }

    Compress-Archive -Path $file.FullName -DestinationPath $destinationPath

    # Print the full file path
    Write-Output "Compressed Lib: $($file.FullName)"
}

Write-Output "Stop Compressing"