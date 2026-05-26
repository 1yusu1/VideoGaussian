param(
    [string]$DepthAnythingRepo = $env:DEPTH_ANYTHING_REPO,
    [string]$ModelDir = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    [string]$InputVideo = $env:VIDEO_PATH,
    [string]$ExportDir = $env:DA3_OUTPUT_DIR,
    [double]$Fps = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($DepthAnythingRepo)) {
    throw "DepthAnythingRepo is required. Pass -DepthAnythingRepo or set DEPTH_ANYTHING_REPO."
}
if ([string]::IsNullOrWhiteSpace($InputVideo)) {
    throw "InputVideo is required. Pass -InputVideo or set VIDEO_PATH."
}
if ([string]::IsNullOrWhiteSpace($ExportDir)) {
    throw "ExportDir is required. Pass -ExportDir or set DA3_OUTPUT_DIR."
}

Push-Location $DepthAnythingRepo
try {
    da3 video $InputVideo `
        --model-dir $ModelDir `
        --fps $Fps `
        --export-dir $ExportDir `
        --export-format mini_npz-glb-depth_vis
}
finally {
    Pop-Location
}
