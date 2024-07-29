#!/usr/bin/env bash

__wrap__() {

  PIXI_BIN="$HOME/.pixi/bin"
  PIPX_BIN="$HOME/.local/bin"
  PATH=$PIXI_BIN:$PIPX_BIN:$PATH

  pixi-installed() {
    if hash pixi 2>/dev/null; then
      return 0
    else
      return 1
    fi
  }

  pixi-update() {
    if pixi self-update; then
      return 0
    else
      return 1
    fi
  }

  pixi-global-install() {
    if wget -qO- https://pixi.sh/install.sh | bash; then
      return 0
    else
      return 1
    fi
  }

  pipx-installed() {
    if hash pipx 2>/dev/null; then
      return 0
    else
      return 1
    fi
  }

  pipx-global-install() {
    if pixi global install pipx; then
      return 0
    else
      return 1
    fi
  }

  keyring-installed() {
    if hash keyring 2>/dev/null; then
      return 0
    else
      return 1
    fi
  }

  pipx-install-keyring() {
    if pipx install keyring; then
      return 0
    else
      return 1
    fi
  }

  pipx-google-artifact-registry-injected() {
    local backends=""
    backends=$(keyring --list-backends)
    if echo "$backends" | grep -q "keyrings.gauth.GooglePythonAuth"; then
      return 0
    else
      return 1
    fi
  }

  pipx-inject-google-artifact-registry() {
    if pipx inject keyring \
      keyrings.google-artifactregistry-auth \
      --index-url https://pypi.org/simple \
      --force; then
      return 0
    else
      return 1
    fi
  }

  gcloud-installed() {
    if hash gcloud 2>/dev/null; then
      return 0
    else
      return 1
    fi
  }

  gcloud-global-install() {
    if pixi global install google-cloud-sdk; then
      return 0
    else
      return 1
    fi
  }

  check_gcloud_auth_login() {
    ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    if [ -z "$ACTIVE_ACCOUNT" ]; then
      return 1
    elif [[ $ACTIVE_ACCOUNT == *"@developer.gserviceaccount.com" ]]; then
      return 1
    else
      return 0
    fi
  }

  rerun-installed() {
    if hash rerun 2>/dev/null; then
      return 0
    else
      return 1
    fi
  }

  rerun-global-install() {
    if pixi global install rerun-sdk; then
      return 0
    else
      return 1
    fi
  }

  pre-commit-installed() {
    if hash pre-commit 2>/dev/null; then
      return 0
    else
      return 1
    fi
  }

  pre-commit-global-install() {
    if pixi global install pre-commit; then
      return 0
    else
      return 1
    fi
  }

  pre-commit-install-hooks() {
    if pre-commit install; then
      return 0
    else
      return 1
    fi
  }

  init-dev-environment() {
    local v=""
    local p=""
    local platform=""

    platform=$(uname -sm)
    printf "installing b3d environment on %s\n\n" "$platform"

    echo "checking pipx..."
    if ! pipx-installed; then
      echo "installing pipx..."
      if ! pipx-global-install; then
        echo "couldn't install pipx"
        exit 1
      fi
    fi
    pipx ensurepath &>/dev/null
    v=$(pipx --version)
    p=$(which pipx)
    printf "  ✓ pipx %s installed (%s)\n\n" "$v" "$p"

    echo "checking keyring..."
    if ! pipx-install-keyring; then
      echo "pipx couldn't install keyring"
      exit 1
    fi
    p=$(which keyring)
    printf "  ✓ keyring installed (%s)\n\n" "$p"

    echo "checking google-artifact-registry-auth keyring backend..."
    if ! pipx-google-artifact-registry-injected; then
      echo "injecting google-artifact-registry-auth backend..."
      if ! pipx-inject-google-artifact-registry; then
        echo "pipx couldn't inject google artifact registry keyring"
        exit 1
      fi
    fi
    printf "   ✓ google artifact registry backend injected\n\n"

    echo "checking gcloud..."
    if ! gcloud-installed; then
      echo "installing gcloud..."
      if ! gcloud-global-install; then
        echo "gcloud install failed"
        exit 1
      fi
    fi
    v=$(gcloud --version)
    p=$(which gcloud)
    printf "  ✓ gcloud %s installed (%s)\n\n" "$v" "$p"

    gcloud auth login --update-adc --force
    printf "→ you are authenticated to google artifact registry \o/\n\n"

    echo "checking rerun-sdk..."
    if ! rerun-installed; then
      echo "installing rerun-sdk..."
      if ! rerun-global-install; then
        echo "rerun-sdk install failed"
        exit 1
      fi
    fi
    v=$(rerun --version)
    p=$(which rerun)
    printf "  ✓ rerun-sdk %s installed (%s)\n\n" "$v" "$p"

    echo "checking pre-commit..."
    if ! pre-commit-installed; then
      echo "installing pre-commit..."
      if ! pre-commit-global-install; then
        echo "pre-commit not installed"
        exit 1
      else
        echo "installing hooks in .pre-commit-config.yaml..."
        if ! pre-commit-install-hooks; then
          echo "installing pre-commit hooks failed"
          exit 1
        fi
      fi
    fi
    printf "  ✓ pre-commit hooks installed\n\n"

    echo "installing b3d environments and dependencies"
    pixi install --locked

    if hash nvidia-smi 2>/dev/null; then
      pixi install --environment gpu --locked
    else
      pixi install --environment cpu --locked
    fi

    echo "done"
  }

  init-dev-environment
}
__wrap__
