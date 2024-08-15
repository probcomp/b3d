# experimental win-64 installer for powershell
# ............................................

# updates shell config creating one if it doesn't exist
function Update-Shell {
  param (
    [string]$File,
    [string]$Line
  )

  if (-not (Test-Path -Path $File)) {
    New-Item -ItemType File -Path $File -Force | Out-Null
  }

  if (-not (Select-String -Pattern "^$Line$" -Path $File -Quiet)) {
    Write-Output "Updating '$File'"
    Add-Content -Path $File -Value $Line
  }
}

if (Test-Path -Path ".\b3d") {
  Write-Output "The 'b3d' repo directory already exists."
  return
}

# env variables
$env:B3D_BRANCH = "eightysteele/win-64-test"
$env:ADC_FILE_LOCAL="$HOME\AppData\Roaming\gcloud\application_default_credentials.json"

# install pixi
iwr -useb https://pixi.sh/install.ps1 | iex

# add pixi to path
Update-Shell -File $PROFILE -Line '$env:PATH = "$HOME\.pixi\bin;$env:PATH"'

# add pixi autocomplete
Update-Shell -Fcd ile $PROFILE -Line '(& pixi completion --shell powershell) | Out-String | Invoke-Expression'

# add pipx bin to path
Update-Shell -File $PROFILE -Line '$env:PATH = "$HOME\.local\bin;$env:PATH"'

# add USER variable
Update-Shell -File $PROFILE -Line '$USER = $env:USERNAME'

# reload profile
. $PROFILE

# install pipx and keyring (for gcloud auth)
pixi global install pipx
pipx install keyring --force
pipx inject keyring keyrings.google-artifactregistry-auth --index-url https://pypi.org/simple --force

# install python (needed by gcloud), git, and gh
pixi global install python git gh

# authenticate gh if needed
if (-not (gh auth status)) {
    gh auth status
    gh auth login --web
}

# authenticate gcloud
if (-not (Test-Path $env:ADC_FILE_LOCAL)) {
    gcloud auth login --update-adc --force
}

# clone b3d and checkout the branch
gh repo clone probcomp/b3d
cd b3d
git checkout $env:B3D_BRANCH
