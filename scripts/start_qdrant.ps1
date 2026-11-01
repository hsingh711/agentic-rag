# Start the local Qdrant server (tools/qdrant/qdrant.exe) with all storage inside this project.
# Usage (from the project root):  powershell -File scripts/start_qdrant.ps1
# The server listens on http://localhost:6333. Stop it with Ctrl+C or by ending the qdrant process.

$projectRoot = Split-Path -Parent $PSScriptRoot
$storageDir = Join-Path $projectRoot "data\qdrant_storage"
New-Item -ItemType Directory -Force $storageDir | Out-Null

$env:QDRANT__STORAGE__STORAGE_PATH = $storageDir
$env:QDRANT__STORAGE__SNAPSHOTS_PATH = Join-Path $storageDir "snapshots"
$env:QDRANT__TELEMETRY_DISABLED = "true"

Set-Location (Join-Path $projectRoot "tools\qdrant")
& .\qdrant.exe
