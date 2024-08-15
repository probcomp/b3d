# experimental win-64 installer for powershell
# ............................................

<#
.SYNOPSIS
    B3D install script.
.DESCRIPTION
    This script is used to install B3D on Windows from the command line.
.PARAMETER PixiVersion
    Specifies the version of Pixi to install.
    The default value is 'latest'. You can also specify it by setting the
    environment variable 'PIXI_VERSION'.
.PARAMETER PixiHome
    Specifies Pixi's home directory.
    The default value is '$Env:USERPROFILE\.pixi'. You can also specify it by
    setting the environment variable 'PIXI_HOME'.
.PARAMETER NoPathUpdate
    If specified, the script will not update the PATH environment variable.
.LINK
    https://github.com/probcomp/b3d
.NOTES
    Version: v0.0.0
#>
param (
    [string] $PixiVersion = 'latest',
    [string] $PixiHome = "$Env:USERPROFILE\.pixi",
    [switch] $NoPathUpdate
)

Set-StrictMode -Version Latest

function Publish-Env {
    if (-not ("Win32.NativeMethods" -as [Type])) {
        Add-Type -Namespace Win32 -Name NativeMethods -MemberDefinition @"
[DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
public static extern IntPtr SendMessageTimeout(
    IntPtr hWnd, uint Msg, UIntPtr wParam, string lParam,
    uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);
"@
    }

    $HWND_BROADCAST = [IntPtr] 0xffff
    $WM_SETTINGCHANGE = 0x1a
    $result = [UIntPtr]::Zero

    [Win32.Nativemethods]::SendMessageTimeout($HWND_BROADCAST,
        $WM_SETTINGCHANGE,
        [UIntPtr]::Zero,
        "Environment",
        2,
        5000,
        [ref] $result
    ) | Out-Null
}

function Write-Env {
    param(
        [String] $name,
        [String] $val,
        [Switch] $global
    )

    $RegisterKey = if ($global) {
        Get-Item -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager'
    } else {
        Get-Item -Path 'HKCU:'
    }

    $EnvRegisterKey = $RegisterKey.OpenSubKey('Environment', $true)
    if ($null -eq $val) {
        $EnvRegisterKey.DeleteValue($name)
    } else {
        $RegistryValueKind = if ($val.Contains('%')) {
            [Microsoft.Win32.RegistryValueKind]::ExpandString
        } elseif ($EnvRegisterKey.GetValue($name)) {
            $EnvRegisterKey.GetValueKind($name)
        } else {
            [Microsoft.Win32.RegistryValueKind]::String
        }
        $EnvRegisterKey.SetValue($name, $val, $RegistryValueKind)
    }
    Publish-Env
}

function Get-Env {
    param(
        [String] $name,
        [Switch] $global
    )

    $RegisterKey = if ($global) {
        Get-Item -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager'
    } else {
        Get-Item -Path 'HKCU:'
    }

    $EnvRegisterKey = $RegisterKey.OpenSubKey('Environment')
    $RegistryValueOption = [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames
    $EnvRegisterKey.GetValue($name, $null, $RegistryValueOption)
}

if ($Env:PIXI_VERSION) {
    $PixiVersion = $Env:PIXI_VERSION
}

if ($Env:PIXI_HOME) {
    $PixiHome = $Env:PIXI_HOME
}

if ($Env:PIXI_NO_PATH_UPDATE) {
    $NoPathUpdate = $true
}

# Repository name
$REPO = 'prefix-dev/pixi'
$ARCH = 'x86_64'
$PLATFORM = 'pc-windows-msvc'

$BINARY = "pixi-$ARCH-$PLATFORM"

if ($PixiVersion -eq 'latest') {
    $DOWNLOAD_URL = "https://github.com/$REPO/releases/latest/download/$BINARY.zip"
} else {
    $DOWNLOAD_URL = "https://github.com/$REPO/releases/download/$PixiVersion/$BINARY.zip"
}

$BinDir = Join-Path $PixiHome 'bin'

Write-Host "This script will automatically download and install Pixi ($PixiVersion) for you."
Write-Host "Getting it from this url: $DOWNLOAD_URL"
Write-Host "The binary will be installed into '$BinDir'"

$TEMP_FILE = [System.IO.Path]::GetTempFileName()

try {
    Invoke-WebRequest -Uri $DOWNLOAD_URL -OutFile $TEMP_FILE

    # Create the install dir if it doesn't exist
    if (!(Test-Path -Path $BinDir)) {
        New-Item -ItemType Directory -Path $BinDir | Out-Null
    }

    $ZIP_FILE = $TEMP_FILE + ".zip"
    Rename-Item -Path $TEMP_FILE -NewName $ZIP_FILE

    # Extract pixi from the downloaded zip file
    Expand-Archive -Path $ZIP_FILE -DestinationPath $BinDir -Force
} catch {
    Write-Host "Error: '$DOWNLOAD_URL' is not available or failed to download"
    exit 1
} finally {
    Remove-Item -Path $ZIP_FILE
}

# Add pixi to PATH if the folder is not already in the PATH variable
if (!$NoPathUpdate) {
    $PATH = Get-Env 'PATH'
    if ($PATH -notlike "*$BinDir*") {
        Write-Output "Adding $BinDir to PATH"
        # For future sessions
        Write-Env -name 'PATH' -val "$BinDir;$PATH"
        # For current session
        $Env:PATH = "$BinDir;$PATH"
        Write-Output "You may need to restart your shell"
    } else {
        Write-Output "$BinDir is already in PATH"
    }
} else {
    Write-Output "You may need to update your PATH manually to use pixi"
}





if (Test-Path -Path ".\b3d") {
  Write-Output "The 'b3d' repo directory already exists."
  return
}

param (
    [string] $B3D_BRANCH = "eightysteele/win-64-test",
    [string] $ADC_FILE_LOCAL="$Env:USERPROFILE\AppData\Roaming\gcloud\application_default_credentials.json"
    [string] $PipxHome = "$Env:USERPROFILE\.local",

)

function Update-Env {
  param (
      [string]$Name,
      [string]$Value,
      [string]$Message
  )
  Write-Output $Message
  # For future sessions
  Write-Env -name $Name -val $Value
  # For the current session
  $Env:$Name = $Value
}

function Add-Autocomplete {
  if (-not (Test-Path -Path $PROFILE)) {
      Write-Output "Profile file not found, creating a new one."
      New-Item -Path $PROFILE -ItemType File -Force
  } else {
      Write-Output "Profile file found."
  }
  Write-Output "Adding Pixi autocomplete to profile."
  Add-Content -Path $PROFILE -Value '(& pixi completion --shell powershell) | Out-String | Invoke-Expression'
}

$PATH = Get-Env 'PATH'
$PipxBinDir = Join-Path $PipxHome 'bin'

Update-Env -Name 'PATH' -Value "$BinDir;$PATH" -Message "Adding $BinDir to PATH"
Update-Env -Name 'PATH' -Value "$PipxBinDir;$PATH" -Message "Adding $PipxBinDir to PATH"
Update-Env -Name 'USER' -Value "$env:USERNAME" -Message "Adding USER variable"
Write-Output "You may need to restart your shell"

Add_Autocomplete

# reload profile
. $PROFILE

# install pipx and keyring (for gcloud auth)
pixi global install pipx
pipx install keyring --force
pipx inject keyring keyrings.google-artifactregistry-auth --index-url https://pypi.org/simple --force

# install python, git, and gh
pixi global install python git gh

# authenticate gh if needed
if (-not (gh auth status)) {
    gh auth status
    gh auth login --web
}

# authenticate gcloud
if (-not (Test-Path $ADC_FILE_LOCAL)) {
    gcloud auth login --update-adc --force
}

# clone b3d and checkout the branch
gh repo clone probcomp/b3d
cd b3d
git checkout $B3D_BRANCH
cd ..

Write-Output "Done!"
