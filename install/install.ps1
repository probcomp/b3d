# Define paths
$PixiHome = "$Env:USERPROFILE\.pixi"
$PixiBinDir = Join-Path $PixiHome 'bin'
$PipxHome = "$Env:USERPROFILE\.local"
$PipxBinDir = Join-Path $PipxHome 'bin'

# Function to update PATH
$script:addedPaths = @()

function Update-PathPermanently {
    param (
        [string]$NewPath
    )
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$NewPath*") {
        $newUserPath = "$currentPath;$NewPath"
        [Environment]::SetEnvironmentVariable("PATH", $newUserPath, "User")
        $Env:PATH = "$Env:PATH;$NewPath"

        # Add to our tracking array
        $script:addedPaths += $NewPath

        Write-Output "Added to PATH: $NewPath"
    } else {
        Write-Output "PATH already contains: $NewPath"
    }
}

function Add-PathToProfile {
    param(
        [string[]]$Paths
    )

    if ($Paths.Count -eq 0) {
        Write-Host "No paths to add to the profile."
        return
    }

    $profileContent = @"
# Ensure critical directories are in PATH
`$criticalPaths = @(
$($Paths | ForEach-Object { "    `"$_`"" } | Join-String -Separator ",`n")
)

foreach (`$path in `$criticalPaths) {
    if (`$env:PATH -notlike "*`$path*") {
        `$env:PATH += ";`$path"
    }
}
"@

    if (Test-Path $PROFILE) {
        $currentContent = Get-Content $PROFILE -Raw
        if ($currentContent -notmatch [regex]::Escape($profileContent)) {
            Add-Content -Path $PROFILE -Value "`n$profileContent"
            Write-Host "Profile updated with PATH checks."
        } else {
            Write-Host "PATH checks already present in profile."
        }
    } else {
        Set-Content -Path $PROFILE -Value $profileContent
        Write-Host "Profile created with PATH checks."
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
    return ((Get-Process -Name "mstsc" -ErrorAction SilentlyContinue) -or
            (Get-ChildItem -Path Env:\ | Where-Object { $_.Name -like "*SESSIONNAME*" -and $_.Value -like "*RDP*" }) -or
            ($env:TERM_PROGRAM -eq "vscode") -or
            [bool](Get-CimInstance -ClassName Win32_SystemAccount -Filter "Name = 'NETWORK SERVICE'" -ErrorAction SilentlyContinue))
}

# Set up GitHub CLI
if (-not (& gh auth status 2>$null)) {
    Write-Output "Authenticating with GitHub..."

    $browserAvailable = Test-BrowserAvailable
    $isRemoteSession = Test-RemoteSession

    if ($browserAvailable -and -not $isRemoteSession) {
        # Local session with browser available
        & gh auth login --web
    } else {
        # Remote session or no browser available
        Write-Output "It seems you're in a remote session or no browser is available."
        Write-Output "You can authenticate using a device code or a token."
        $authMethod = Read-Host "Choose authentication method: [D]evice code or [T]oken"

        if ($authMethod -eq 'D' -or $authMethod -eq 'd') {
            & gh auth login
        } elseif ($authMethod -eq 'T' -or $authMethod -eq 't') {
            & gh auth login --with-token
            Write-Output "Please paste your GitHub personal access token when prompted."
        } else {
            Write-Output "Invalid choice. Skipping GitHub authentication. Please run 'gh auth login' manually later."
        }
    }
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
        Write-Output "gcloud auth login --no-browser --update-adc --force"

        $proceed = Read-Host "Do you want to proceed with browser-less authentication now? (y/n)"
        if ($proceed -eq 'y') {
            & gcloud auth login --no-browser --update-adc --force
        } else {
            Write-Output "Skipping Google Cloud authentication. Please run the command manually later."
        }
    }
}

function Check-And-Handle-B3DDirectory {
    $b3dPath = Join-Path $PWD "b3d"
    if (Test-Path $b3dPath) {
        Write-Host "A 'b3d' directory already exists in the current location."
        $decision = Read-Host "Do you want to delete it? (y/n)"
        if ($decision -eq 'y') {
            try {
                Remove-Item -Path $b3dPath -Recurse -Force
                Write-Host "The 'b3d' directory has been deleted."
            }
            catch {
                Write-Host "Error deleting the 'b3d' directory: $_"
                Write-Host "Please delete the directory manually and run the script again."
                exit
            }
        }
        else {
            Write-Host "The 'b3d' directory was not deleted. Exiting the script."
            exit
        }
    }
    else {
        Write-Host "No existing 'b3d' directory found. Proceeding with the script."
    }
}

Check-And-Handle-B3DDirectory

# Clone and checkout b3d repository
$B3D_BRANCH = "eightysteele/win-64-test"
if (-not (Test-Path "b3d")) {
    Write-Output "Cloning b3d repository..."
    & gh repo clone probcomp/b3d
    Set-Location b3d
    & git checkout $B3D_BRANCH
    Set-Location ..
}

if ($script:addedPaths.Count -gt 0) {
    Write-Host "Setup complete. The following paths have been added to your system PATH:"
    $script:addedPaths | ForEach-Object { Write-Host " - $_" }

    Write-Host "`nTo ensure these paths are always available in your PowerShell sessions, you can add them to your profile."
    $addProfileUpdates = Read-Host "Would you like to update your PowerShell profile? (y/n)"

    if ($addProfileUpdates -eq 'y') {
        Add-PathToProfile -Paths $script:addedPaths
        Write-Host "To apply these changes to your current session, please run:"
        Write-Host ". `$PROFILE"
    } else {
        Write-Host "No changes made to your PowerShell profile."
        Write-Host "If you want to add these paths to your profile later, you can run:"
        Write-Host "Add-PathToProfile -Paths '$($script:addedPaths -join "','")'"
    }
} else {
    Write-Host "Setup complete. No new paths were added to your system PATH."
}

Write-Output "PATH updates have been checked and added to your PowerShell profile if necessary."
Write-Output "To ensure all changes take effect, please restart your PowerShell session or run:"
Write-Output ". `$PROFILE"
