# THANK YOU https://github.com/ptheywood/cuda-cmake-github-actions

## -------------------
## Constants
## -------------------

# Dictionary of known cuda versions and thier download URLS, which do not follow a consistent pattern
# From 11.0, the download url/toolkit version is separate from the cudart version.
# Releases since 11.5.1 (including 11.4.4) use `windows` rather than `win10` in the uri, due to windows 11 inclusion
$CUDA_KNOWN_URLS = @{
    "8.0.44"   = "https://developer.nvidia.com/compute/cuda/8.0/Prod/network_installers/cuda_8.0.44_win10_network-exe";
    "8.0.61"   = "https://developer.nvidia.com/compute/cuda/8.0/Prod2/network_installers/cuda_8.0.61_win10_network-exe";
    "9.0.176"  = "https://developer.nvidia.com/compute/cuda/9.0/Prod/network_installers/cuda_9.0.176_win10_network-exe";
    "9.1.85"   = "https://developer.nvidia.com/compute/cuda/9.1/Prod/network_installers/cuda_9.1.85_win10_network";
    "9.2.148"  = "https://developer.nvidia.com/compute/cuda/9.2/Prod2/network_installers2/cuda_9.2.148_win10_network";
    "10.0.130" = "https://developer.nvidia.com/compute/cuda/10.0/Prod/network_installers/cuda_10.0.130_win10_network";
    "10.1.105" = "https://developer.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.105_win10_network.exe";
    "10.1.168" = "https://developer.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.168_win10_network.exe";
    "10.1.243" = "https://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe";
    "10.2.89"  = "https://developer.download.nvidia.com/compute/cuda/10.2/Prod/network_installers/cuda_10.2.89_win10_network.exe";
    "11.0.1" = "https://developer.download.nvidia.com/compute/cuda/11.0.1/network_installers/cuda_11.0.1_win10_network.exe";
    "11.0.2" = "https://developer.download.nvidia.com/compute/cuda/11.0.2/network_installers/cuda_11.0.2_win10_network.exe";
    "11.0.3" = "https://developer.download.nvidia.com/compute/cuda/11.0.3/network_installers/cuda_11.0.3_win10_network.exe";
    "11.1.0" = "https://developer.download.nvidia.com/compute/cuda/11.1.0/network_installers/cuda_11.1.0_win10_network.exe";
    "11.1.1" = "https://developer.download.nvidia.com/compute/cuda/11.1.1/network_installers/cuda_11.1.1_win10_network.exe";
    "11.2.0" = "https://developer.download.nvidia.com/compute/cuda/11.2.0/network_installers/cuda_11.2.0_win10_network.exe";
    "11.2.1" = "https://developer.download.nvidia.com/compute/cuda/11.2.1/network_installers/cuda_11.2.1_win10_network.exe";
    "11.2.2" = "https://developer.download.nvidia.com/compute/cuda/11.2.2/network_installers/cuda_11.2.2_win10_network.exe";
    "11.3.0" = "https://developer.download.nvidia.com/compute/cuda/11.3.0/network_installers/cuda_11.3.0_win10_network.exe";
    "11.3.1" = "https://developer.download.nvidia.com/compute/cuda/11.3.1/network_installers/cuda_11.3.1_win10_network.exe";
    "11.4.0" = "https://developer.download.nvidia.com/compute/cuda/11.4.0/network_installers/cuda_11.4.0_win10_network.exe";
    "11.4.1" = "https://developer.download.nvidia.com/compute/cuda/11.4.1/network_installers/cuda_11.4.1_win10_network.exe";
    "11.4.2" = "https://developer.download.nvidia.com/compute/cuda/11.4.2/network_installers/cuda_11.4.2_win10_network.exe";
    "11.4.3" = "https://developer.download.nvidia.com/compute/cuda/11.4.3/network_installers/cuda_11.4.3_win10_network.exe";
    "11.4.4" = "https://developer.download.nvidia.com/compute/cuda/11.4.4/network_installers/cuda_11.4.4_windows_network.exe";
    "11.5.0" = "https://developer.download.nvidia.com/compute/cuda/11.5.0/network_installers/cuda_11.5.0_win10_network.exe";
    "11.5.1" = "https://developer.download.nvidia.com/compute/cuda/11.5.1/network_installers/cuda_11.5.1_windows_network.exe";
    "11.5.2" = "https://developer.download.nvidia.com/compute/cuda/11.5.2/network_installers/cuda_11.5.2_windows_network.exe";
    "11.6.0" = "https://developer.download.nvidia.com/compute/cuda/11.6.0/network_installers/cuda_11.6.0_windows_network.exe";
    "11.6.1" = "https://developer.download.nvidia.com/compute/cuda/11.6.1/network_installers/cuda_11.6.1_windows_network.exe";
    "11.6.2" = "https://developer.download.nvidia.com/compute/cuda/11.6.2/network_installers/cuda_11.6.2_windows_network.exe";
    "11.7.0" = "https://developer.download.nvidia.com/compute/cuda/11.7.0/network_installers/cuda_11.7.0_windows_network.exe";
    "11.7.1" = "https://developer.download.nvidia.com/compute/cuda/11.7.1/network_installers/cuda_11.7.1_windows_network.exe";
    "11.8.0" = "https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe";
    "12.0.0" = "https://developer.download.nvidia.com/compute/cuda/12.0.0/network_installers/cuda_12.0.0_windows_network.exe"
}

# @todo - change this to be based on _MSC_VER intead, or invert it to be CUDA keyed instead
$VISUAL_STUDIO_MIN_CUDA = @{
    "2022" = "11.6.0";
    "2019" = "10.1";
    "2017" = "10.0"; # Depends on which version of 2017! 9.0 to 10.0 depending on version
    "2015" = "8.0";  # Might support older, unsure. Depracated as of 11.1, unsupported in 11.2
}

# cuda_runtime.h is in nvcc <= 10.2, but cudart >= 11.0
# @todo - make this easier to vary per CUDA version.
$CUDA_PACKAGES_IN = @(
    "nvcc";
    "visual_studio_integration";
    "curand_dev";
    "nvrtc_dev";
    "cudart";
    "thrust";
)

## -------------------
## Select CUDA version
## -------------------

# Get the cuda version from the environment as env:cuda.
$CUDA_VERSION_FULL = $env:cuda
# Make sure CUDA_VERSION_FULL is set and valid, otherwise error.

# Validate CUDA version, extracting components via regex
$cuda_ver_matched = $CUDA_VERSION_FULL -match "^(?<major>[1-9][0-9]*)\.(?<minor>[0-9]+)\.(?<patch>[0-9]+)$"
if(-not $cuda_ver_matched){
    Write-Output "Invalid CUDA version specified, <major>.<minor>.<patch> required. '$CUDA_VERSION_FULL'."
    exit 1
}
$CUDA_MAJOR=$Matches.major
$CUDA_MINOR=$Matches.minor
$CUDA_PATCH=$Matches.patch

## ---------------------------
## Visual studio support check
## ---------------------------
# Exit if visual studio is too new for the cuda version.
$VISUAL_STUDIO = $env:visual_studio.trim()
if ($VISUAL_STUDIO.length -ge 4) {
$VISUAL_STUDIO_YEAR = $VISUAL_STUDIO.Substring($VISUAL_STUDIO.Length-4)
    if ($VISUAL_STUDIO_YEAR.length -eq 4 -and $VISUAL_STUDIO_MIN_CUDA.containsKey($VISUAL_STUDIO_YEAR)){
        $MINIMUM_CUDA_VERSION = $VISUAL_STUDIO_MIN_CUDA[$VISUAL_STUDIO_YEAR]
        if ([version]$CUDA_VERSION_FULL -lt [version]$MINIMUM_CUDA_VERSION) {
            Write-Output "Error: Visual Studio $($VISUAL_STUDIO_YEAR) requires CUDA >= $($MINIMUM_CUDA_VERSION)"
            exit 1
        }
    }
} else {
    Write-Output "Warning: Unknown Visual Studio Version. CUDA version may be insufficient."
}

## ------------------------------------------------
## Select CUDA packages to install from environment
## ------------------------------------------------

$CUDA_PACKAGES = ""
Foreach ($package in $CUDA_PACKAGES_IN) {
    # Make sure the correct package name is used for nvcc.
    if($package -eq "nvcc" -and [version]$CUDA_VERSION_FULL -lt [version]"9.1"){
        $package="compiler"
    } elseif($package -eq "compiler" -and [version]$CUDA_VERSION_FULL -ge [version]"9.1") {
        $package="nvcc"
    } elseif($package -eq "thrust" -and [version]$CUDA_VERSION_FULL -lt [version]"11.3") {
        # Thrust is a package from CUDA 11.3, otherwise it should be skipped.
        continue
    }
    $CUDA_PACKAGES += " $($package)_$($CUDA_MAJOR).$($CUDA_MINOR)"
}
echo "$($CUDA_PACKAGES)"
## -----------------
## Prepare download
## -----------------

# Select the download link if known, otherwise have a guess.
$CUDA_REPO_PKG_REMOTE=""
$CUDA_REPO_PKG_LOCAL=""
if($CUDA_KNOWN_URLS.containsKey($CUDA_VERSION_FULL)){
    $CUDA_REPO_PKG_REMOTE=$CUDA_KNOWN_URLS[$CUDA_VERSION_FULL]
} else{
    # Guess what the url is given the most recent pattern (at the time of writing, 10.1)
    Write-Output "note: URL for CUDA ${$CUDA_VERSION_FULL} not known, estimating."
    if([version]$CUDA_VERSION_FULL -ge [version]"11.5.1"){
        $CUDA_REPO_PKG_REMOTE="https://developer.download.nvidia.com/compute/cuda/$($CUDA_MAJOR).$($CUDA_MINOR)/Prod/network_installers/cuda_$($CUDA_VERSION_FULL)_windows_network.exe"
    } else {
        $CUDA_REPO_PKG_REMOTE="https://developer.download.nvidia.com/compute/cuda/$($CUDA_MAJOR).$($CUDA_MINOR)/Prod/network_installers/cuda_$($CUDA_VERSION_FULL)_win10_network.exe"
    }
}
if([version]$CUDA_VERSION_FULL -ge [version]"11.5.1"){
    $CUDA_REPO_PKG_LOCAL="cuda_$($CUDA_VERSION_FULL)_windows_network.exe"
} else {
    $CUDA_REPO_PKG_LOCAL="cuda_$($CUDA_VERSION_FULL)_win10_network.exe"
}

## ------------
## Install CUDA
## ------------

# Get CUDA network installer, retrying upto N times.
Write-Output "Downloading CUDA Network Installer for $($CUDA_VERSION_FULL) from: $($CUDA_REPO_PKG_REMOTE)"

$downloaded = $false
$download_attempt = 0
$download_attempt_delay = 30
$download_attempts_max = 5

while (-not $downloaded) {
    Invoke-WebRequest $CUDA_REPO_PKG_REMOTE -OutFile $CUDA_REPO_PKG_LOCAL | Out-Null
    $download_attempt++
    # If download succeeded, break out the loop.
    if(Test-Path -Path $CUDA_REPO_PKG_LOCAL){
        Write-Output "Downloading Complete"
        $downloaded=$true
    } else {
        # If downlaod failed, either wait and try again, or give up and error.
        if ($download_attempt -le $download_attempts_max) {
            Write-Output "Error: Failed to download $($CUDA_REPO_PKG_LOCAL) (attempt $($download_attempt)/$($download_attempts_max)). Retrying."
            # Sleep for a number of seconds.
            Start-Sleep $download_attempt_delay
        } else {
            Write-Output "Error: Failed to download $($CUDA_REPO_PKG_LOCAL) after $($download_attempts_max) attempts. Aborting."
            # Abort the script.
            exit 1
        }
    }
}

# Invoke silent install of CUDA (via network installer)
Write-Output "Installing CUDA $($CUDA_VERSION_FULL). Subpackages $($CUDA_PACKAGES)"
Start-Process -Wait -FilePath .\"$($CUDA_REPO_PKG_LOCAL)" -ArgumentList "-s $($CUDA_PACKAGES)"

# Check the return status of the CUDA installer.
if (!$?) {
    Write-Output "Error: CUDA installer reported error. $($LASTEXITCODE)"
    exit 1
}

# Store the CUDA_PATH in the environment for the current session, to be forwarded in the action.
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$($CUDA_MAJOR).$($CUDA_MINOR)"
$CUDA_PATH_VX_Y = "CUDA_PATH_V$($CUDA_MAJOR)_$($CUDA_MINOR)"
# Set environmental variables in this session
$env:CUDA_PATH = "$($CUDA_PATH)"
$env:CUDA_PATH_VX_Y = "$($CUDA_PATH_VX_Y)"
Write-Output "CUDA_PATH $($CUDA_PATH)"
Write-Output "CUDA_PATH_VX_Y $($CUDA_PATH_VX_Y)"

# PATH needs updating elsewhere, anything in here won't persist.
# Append $CUDA_PATH/bin to path.
# Set CUDA_PATH as an environmental variable

# If executing on github actions, emit the appropriate echo statements to update environment variables
if (Test-Path "env:GITHUB_ACTIONS") {
    # Set paths for subsequent steps, using $env:CUDA_PATH
    echo "Adding CUDA to CUDA_PATH, CUDA_PATH_X_Y and PATH"
    echo "CUDA_PATH=$env:CUDA_PATH" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    echo "$env:CUDA_PATH_VX_Y=$env:CUDA_PATH" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    echo "$env:CUDA_PATH/bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
}