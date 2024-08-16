# Define paths
$PixiHome = "$Env:USERPROFILE\.pixi"
$PixiBinDir = Join-Path $PixiHome 'bin'
$PipxHome = "$Env:USERPROFILE\.local"
$PipxBinDir = Join-Path $PipxHome 'bin'

# Function to update PATH
function Update-PathPermanently {
    param (
        [string]$NewPath
    )
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$NewPath*") {
        $newUserPath = "$currentPath;$NewPath"
        [Environment]::SetEnvironmentVariable("PATH", $newUserPath, "User")
        $Env:PATH = "$Env:PATH;$NewPath"
    }
}

# Install Pixi
if (-not (Test-Path $PixiBinDir)) {
    Write-Output "Installing Pixi..."
    Invoke-Expression (Invoke-WebRequest -useb https://pixi.sh/install.ps1)
}

# Update PATH for Pixi
Update-PathPermanently $PixiBinDir

# Check if PowerShell profile exists, create if it doesn't
if (-not (Test-Path -Path $PROFILE)) {
    Write-Output "PowerShell profile does not exist. Creating it..."
    New-Item -ItemType File -Path $PROFILE -Force
}

# Add Pixi autocomplete to user profile
$AutoCompleteCommand = '(& pixi completion --shell powershell) | Out-String | Invoke-Expression'
if (-not (Select-String -Path $PROFILE -Pattern "pixi completion" -Quiet)) {
    Add-Content -Path $PROFILE -Value $AutoCompleteCommand
    # Load autocomplete in current session
    Invoke-Expression $AutoCompleteCommand
}

# Refresh PATH to include Pixi
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Install Pipx using Pixi
Write-Output "Installing Pipx..."
& pixi global install pipx

# Update PATH for Pipx
Update-PathPermanently $PipxBinDir

# Refresh PATH to include Pipx
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Install keyring using Pipx
Write-Output "Installing keyring..."
& pipx install keyring --force
& pipx inject keyring keyrings.google-artifactregistry-auth --index-url https://pypi.org/simple --force

# Install other tools using Pixi
Write-Output "Installing Python, Git, and GitHub CLI..."
& pixi global install python git gh

# Set up GitHub CLI
if (-not (& gh auth status 2>$null)) {
    Write-Output "Authenticating with GitHub..."
    & gh auth login --web
}

# Function to check if a local browser is available
function Test-BrowserAvailable {
    try {
        $key = [Microsoft.Win32.Registry]::ClassesRoot.OpenSubKey("http\shell\open\command", $false)
        $default = $key.GetValue("")
        return $null -ne $default
    }
    catch {
        return $false
    }
}

# Function to check if we're in a remote session
function Test-RemoteSession {
    return (Get-Process -Name "mstsc" -ErrorAction SilentlyContinue) -or [System.Windows.Forms.SystemInformation]::TerminalServerSession
}

# Set up Google Cloud SDK
$ADC_FILE_LOCAL = "$Env:USERPROFILE\AppData\Roaming\gcloud\application_default_credentials.json"
if (-not (Test-Path $ADC_FILE_LOCAL)) {
    Write-Output "Authenticating with Google Cloud..."

    $browserAvailable = Test-BrowserAvailable
    $isRemoteSession = Test-RemoteSession

    if ($browserAvailable -and -not $isRemoteSession) {
        # Local session with browser available
        & gcloud auth login --update-adc --force
    } else {
        # Remote session or no browser available
        Write-Output "It seems you're in a remote session or no browser is available."
        Write-Output "Please use this command to authenticate:"
        Write-Output "gcloud auth login --no-launch-browser --update-adc"

        $proceed = Read-Host "Do you want to proceed with browser-less authentication now? (y/n)"
        if ($proceed -eq 'y') {
            & gcloud auth login --no-launch-browser --update-adc --force
        } else {
            Write-Output "Skipping Google Cloud authentication. Please run the command manually later."
        }
    }
}

# Clone and checkout b3d repository
$B3D_BRANCH = "eightysteele/win-64-test"
if (-not (Test-Path "b3d")) {
    Write-Output "Cloning b3d repository..."
    & gh repo clone probcomp/b3d
    Set-Location b3d
    & git checkout $B3D_BRANCH
    Set-Location ..
}

Write-Output "Setup complete! Please restart your PowerShell session for all changes to take effect."
Write-Output "You can do this by running: powershell -NoExit -Command `"& {Import-Module `$profile}`""
