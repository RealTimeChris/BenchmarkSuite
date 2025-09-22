# Dead Code Detection - BULLETPROOF VERSION
$SourceDirectory = Get-Location
$BuildDirectory = 'build'
$Generator = 'Visual Studio 17 2022'
$TestDeletionFolder = './include/cute_rt_tm'

Write-Host '=== Dead Code Detection - ALL FILES ===' -ForegroundColor Green
Write-Host "Source Directory: $SourceDirectory" -ForegroundColor Yellow
Write-Host "Build Directory: $BuildDirectory" -ForegroundColor Yellow
Write-Host "Testing deletions in: $TestDeletionFolder" -ForegroundColor Yellow
Write-Host ''

if (-not (Test-Path $TestDeletionFolder)) {
    Write-Host "ERROR: Test deletion folder doesn't exist!" -ForegroundColor Red
    exit 1
}

# Get ALL files
$filesToTest = Get-ChildItem -Path $TestDeletionFolder -Recurse -File
$fileCount = $filesToTest.Count
Write-Host "Found $fileCount FILES TO TEST!" -ForegroundColor Green
Write-Host ''

$safeToDelete = 0
$neededFiles = 0
$testNumber = 1

foreach ($file in $filesToTest) {
    $originalPath = $file.FullName
    $fileName = $file.Name
    $relativePath = $originalPath.Substring((Get-Location).Path.Length + 1)
    
    Write-Host "$testNumber / $fileCount - Testing: $relativePath" -ForegroundColor White
    $testNumber++
    
    # Backup and delete
    $backupPath = "$env:TEMP\$fileName.backup"
    Copy-Item $originalPath $backupPath -Force
    Remove-Item $originalPath -Force
    
    Write-Host "  Deleted $fileName, testing build..." -ForegroundColor Gray
    
    # FULL BUILD TEST
    $buildSuccess = $false
    
    # Step 1: Configure - MANUAL QUOTES
    Push-Location $SourceDirectory
    $configArgs = '-G "Visual Studio 17 2022" -S . -B build -DBENCH_TYPE=BENCHMARK'
    $configureProcess = Start-Process -FilePath 'cmake.exe' -ArgumentList $configArgs -Wait -PassThru -NoNewWindow
    if ($configureProcess.ExitCode -eq 0) {
        # Step 2: Clean
        Pop-Location
        Push-Location $BuildDirectory
        $cleanArgs = '--build . --config Release --target clean'
        $cleanProcess = Start-Process -FilePath 'cmake.exe' -ArgumentList $cleanArgs -Wait -PassThru -NoNewWindow
        if ($cleanProcess.ExitCode -eq 0) {
            # Step 3: Build
            $buildArgs = '--build . --config Release'
            $buildProcess = Start-Process -FilePath 'cmake.exe' -ArgumentList $buildArgs -Wait -PassThru -NoNewWindow -RedirectStandardOutput "$env:TEMP\build_out.txt" -RedirectStandardError "$env:TEMP\build_err.txt"
            $exitCode = $buildProcess.ExitCode
            
            # Check output for errors
            $buildOutput = @()
            if (Test-Path "$env:TEMP\build_out.txt") { 
                $buildOutput += Get-Content "$env:TEMP\build_out.txt" 
            }
            if (Test-Path "$env:TEMP\build_err.txt") { 
                $buildOutput += Get-Content "$env:TEMP\build_err.txt" 
            }
            Remove-Item "$env:TEMP\build_out.txt" -ErrorAction SilentlyContinue
            Remove-Item "$env:TEMP\build_err.txt" -ErrorAction SilentlyContinue
            
            $outputText = ($buildOutput | Out-String).ToLower()
            $hasError = $outputText -match 'error|failed|cannot|fatal'
            
            $buildSuccess = ($exitCode -eq 0) -and (-not $hasError)
        }
        Pop-Location
    }
    
    # RESULT
    if ($buildSuccess) {
        Write-Host "  BUILD SUCCEEDED - $fileName is DEAD CODE! (Safe to delete)" -ForegroundColor Green
        Remove-Item $backupPath -Force
        $safeToDelete++
    } else {
        Write-Host "  BUILD FAILED - $fileName is NEEDED!" -ForegroundColor Red
        if ($buildOutput.Count -gt 0) {
            Write-Host "     Last 3 lines:" -ForegroundColor Red
            $buildOutput | Select-Object -Last 3 | ForEach-Object { Write-Host "       $_" -ForegroundColor Red }
        }
        Copy-Item $backupPath $originalPath -Force
        Remove-Item $backupPath -Force
        $neededFiles++
    }
    Write-Host ''
}

Write-Host '=== RESULTS ===' -ForegroundColor Green
Write-Host "SAFE TO DELETE: $safeToDelete files" -ForegroundColor Green
Write-Host "NEEDED: $neededFiles files" -ForegroundColor Red
Write-Host "TOTAL: $fileCount files" -ForegroundColor White
