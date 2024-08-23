#!/usr/bin/env bash

__wrap__() {

  # ............................................................................
  # b3d

  if hash conda 2>/dev/null; then
    conda init >/dev/null 2>&1
    conda deactivate >/dev/null 2>&1
    conda config --set auto_activate_base false
  fi

  GIT_USER_NAME=${GIT_USER_NAME:-}
  GIT_USER_EMAIL=${GIT_USER_EMAIL:-}
  B3D_BRANCH=${B3D_BRANCH:-main}
  PIPX_BIN="$HOME/.local/bin"
  ADC_FILE_LOCAL="$HOME/.config/gcloud/application_default_credentials.json"
  ADC_FILE_REMOTE="$HOME/application_default_credentials.json"
  GCLOUD_AUTH_REMOTE=1
  B3D_REMOTE_MACHINE=1

  if [[ -n $SSH_CONNECTION ]] && [[ -n $SSH_CLIENT ]] && [[ -n $SSH_TTY ]]; then
    B3D_REMOTE_MACHINE=0
  else
    B3D_REMOTE_MACHINE=1
  fi

  config-git() {
    if [[ $B3D_REMOTE_MACHINE -eq 1 ]]; then
      return 0
    fi

    if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
      printf "\n→ updating git config --global user.name \"%s\"" "$GIT_USER_NAME"
      printf "\n→ updating git config --global user.email \"%s\"\n" "$GIT_USER_EMAIL"
      git config --global user.name "$GIT_USER_NAME"
      git config --global user.email "$GIT_USER_EMAIL"
    else
      printf "\n  Oops! Your git config is missing (please set and try again)"
      printf "\n  export GIT_USER_NAME=\"Your Name\""
      printf "\n  export GIT_USER_EMAIL=\"Your Email\"\n\n"
      exit 1
    fi
  }

  # ............................................................................
  # pixi install

  VERSION=${PIXI_VERSION:-latest}
  PIXI_HOME=${PIXI_HOME:-"$HOME/.pixi"}
  BIN_DIR="$PIXI_HOME/bin"

  REPO=prefix-dev/pixi
  PLATFORM=$(uname -s)
  ARCH=${PIXI_ARCH:-$(uname -m)}

  if [[ $PLATFORM == "Darwin" ]]; then
    PLATFORM="apple-darwin"
  elif [[ $PLATFORM == "Linux" ]]; then
    PLATFORM="unknown-linux-musl"
  elif [[ $(uname -o) == "Msys" ]]; then
    PLATFORM="pc-windows-msvc"
  fi

  if [[ $ARCH == "arm64" ]] || [[ $ARCH == "aarch64" ]]; then
    ARCH="aarch64"
  fi

  BINARY="pixi-${ARCH}-${PLATFORM}"
  EXTENSION="tar.gz"
  if [[ $(uname -o) == "Msys" ]]; then
    EXTENSION="zip"
  fi

  if [[ $VERSION == "latest" ]]; then
    DOWNLOAD_URL=https://github.com/${REPO}/releases/latest/download/${BINARY}.${EXTENSION}
  else
    DOWNLOAD_URL=https://github.com/${REPO}/releases/download/${VERSION}/${BINARY}.${EXTENSION}
  fi

  platform=$(uname -sm)

  if [ -d "$PWD/b3d" ]; then
    printf "\nOops! The 'b3d' directory exists! Please rename or remove it, then try again.\n"
    exit 1
  fi

  config-git

  local type
  if [[ $B3D_REMOTE_MACHINE -eq 1 ]]; then
    type="local"
  else
    type="remote"
  fi
  printf "\nInstalling the b3d development environment (%s %s machine)...\n" "$type" "$platform"

  sleep 3

  if ! hash curl 2>/dev/null && ! hash wget 2>/dev/null; then
    echo "error: you need either 'curl' or 'wget' installed for this script."
    exit 1
  fi

  if ! hash tar 2>/dev/null; then
    echo "error: you do not have 'tar' installed which is required for this script."
    exit 1
  fi

  TEMP_FILE=$(mktemp "${TMPDIR:-/tmp}/.pixi_install.XXXXXXXX")

  cleanup() {
    rm -f "$TEMP_FILE"
  }

  trap cleanup EXIT

  printf "\n→ installing pixi in $PIXI_HOME\n\n"

  if hash curl 2>/dev/null; then
    HTTP_CODE=$(curl -SL --progress-bar "$DOWNLOAD_URL" --output "$TEMP_FILE" --write-out "%{http_code}")
    if [[ ${HTTP_CODE} -lt 200 || ${HTTP_CODE} -gt 299 ]]; then
      echo "error: '${DOWNLOAD_URL}' is not available"
      exit 1
    fi
  elif hash wget 2>/dev/null; then
    if ! wget -q --show-progress --output-document="$TEMP_FILE" "$DOWNLOAD_URL"; then
      echo "error: '${DOWNLOAD_URL}' is not available"
      exit 1
    fi
  fi

  # Check that file was correctly created (https://github.com/prefix-dev/pixi/issues/446)
  if [[ ! -s $TEMP_FILE ]]; then
    echo "error: temporary file ${TEMP_FILE} not correctly created."
    echo "       As a workaround, you can try set TMPDIR env variable to directory with write permissions."
    exit 1
  fi

  # Extract pixi from the downloaded file
  mkdir -p "$BIN_DIR"
  if [[ $(uname -o) == "Msys" ]]; then
    unzip "$TEMP_FILE" -d "$BIN_DIR"
  else
    tar -xzf "$TEMP_FILE" -C "$BIN_DIR"
    chmod +x "$BIN_DIR/pixi"
  fi

  update_shell() {
    FILE=$1
    LINE=$2

    # shell update can be suppressed by `PIXI_NO_PATH_UPDATE` env var
    [[ -n ${PIXI_NO_PATH_UPDATE-} ]] && echo "No path update because PIXI_NO_PATH_UPDATE has a value" && return

    # Create the file if it doesn't exist
    if [ -f "$FILE" ]; then
      touch "$FILE"
    fi

    # Append the line if not already present
    if ! grep -Fxq "$LINE" "$FILE"; then
      echo "Updating '${FILE}'"
      echo "$LINE" >>"$FILE"
      echo "Please restart or source your shell."
    fi
  }

  case "$(basename "$SHELL")" in
  bash)
    # Default to bashrc as that is used in non login shells instead of the profile.
    LINE="export PATH=${BIN_DIR}:\$PATH"
    update_shell ~/.bashrc "$LINE"
    ;;

  fish)
    LINE="fish_add_path ${BIN_DIR}"
    update_shell ~/.config/fish/config.fish "$LINE"
    ;;

  zsh)
    LINE="export PATH=${BIN_DIR}:\$PATH"
    update_shell ~/.zshrc "$LINE"
    ;;

  tcsh)
    LINE="set path = ( ${BIN_DIR} \$path )"
    update_shell ~/.tcshrc "$LINE"
    ;;

  *)
    echo "Could not update shell: $(basename "$SHELL")"
    echo "Please permanently add '${BIN_DIR}' to your \$PATH to enable the 'pixi' command."
    ;;
  esac

  # ............................................................................
  # b3d

  is-gauth-injected() {
    local backends=""
    backends=$(keyring --list-backends)
    if echo "$backends" | grep -q "keyrings.gauth.GooglePythonAuth"; then
      return 0
    else
      return 1
    fi
  }

  inject-gauth() {
    if pipx inject keyring \
      keyrings.google-artifactregistry-auth \
      --index-url https://pypi.org/simple \
      --force; then
      return 0
    else
      return 1
    fi
  }
  printf "\n→ updating shell config\n"

  case "$(basename "$SHELL")" in
  bash)
    # Default to bashrc as that is used in non login shells instead of the profile.
    LINE='eval "$(pixi completion --shell bash)"'
    update_shell ~/.bashrc "$LINE"
    LINE='export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"'
    update_shell ~/.bashrc "$LINE"
    ;;

  fish)
    LINE='pixi completion --shell fish | source'
    update_shell ~/.config/fish/config.fish "$LINE"
    LINE='export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"'
    update_shell ~/.config/fish/config.fish "$LINE"
    ;;

  zsh)
    LINE="autoload -Uz compinit"
    update_shell ~/.zshrc "$LINE"
    LINE="compinit"
    update_shell ~/.zshrc "$LINE"
    LINE='eval "$(pixi completion --shell zsh)"'
    update_shell ~/.zshrc "$LINE"
    LINE='export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"'
    update_shell ~/.zshrc "$LINE"
    ;;

  *)
    echo "could not add pixi autocomplete to shell: $(basename "$SHELL")"
    echo "please see: https://pixi.sh/latest/#autocompletion"
    ;;
  esac

  export PATH=$BIN_DIR:$PIPX_BIN:$PATH

  printf "\n→ installing pipx into %s\n\n" "$BIN_DIR"
  pixi global install pipx
  pipx ensurepath &>/dev/null

  printf "\n→ installing keyring into %s\n\n" "$PIPX_BIN"
  pipx install keyring

  printf "\n→ injecting gauth backend into keyring...\n\n"
  if ! is-gauth-injected; then
    inject-gauth
  fi

  printf "\n→ installing git into %s\n\n" "$BIN_DIR"
  pixi global install git

  printf "\n→ installing gh into %s\n\n" "$BIN_DIR"
  pixi global install gh

  if ! hash gcloud 2>/dev/null; then
    printf "\n→ installing gcloud...\n"
    pixi global install google-cloud-sdk
  fi

  # local machine
  if [[ $B3D_REMOTE_MACHINE -eq 1 ]]; then
    printf "\n→ authenticating gh...\n"
    if ! gh auth status; then
      gh auth status
      gh auth login --web
    else
      printf "\n✓ gh authenticated\n"
    fi
    printf "\n→ authenticating gcloud...\n"
    if ! [[ -e $ADC_FILE_LOCAL ]]; then
      printf "  gcp application default credentials not found  %s\n" "$ADC_FILE_LOCAL"
      gcloud auth login --update-adc --force
    else
      printf "✓ gcp application default credentials found  %s\n" "$ADC_FILE_LOCAL"
    fi
  fi

  # remote machine
  if [[ $B3D_REMOTE_MACHINE -eq 0 ]]; then
    # gcloud creds
    if [[ -e $ADC_FILE_REMOTE ]]; then
      cp -v "$ADC_FILE_REMOTE" "$HOME/.config/gcloud/"
      rm "$ADC_FILE_REMOTE"
      GCLOUD_AUTH_REMOTE=0
    elif ! [[ -e "$HOME/.config/gcloud/application_default_credentials.json" ]]; then
      echo "remote gcloud creds not found $ADC_FILE_REMOTE"
      GCLOUD_AUTH_REMOTE=1
    else
      GCLOUD_AUTH_REMOTE=0
    fi
    # gh creds
    if ! gh auth status; then
      gh auth login
    fi
  fi

  printf "\n→ installing pre-commit into %s\n\n" "$BIN_DIR"
  pixi global install pre-commit

  B3D_ROOT="$PWD/b3d"

  printf "\n→ cloning probcomp/b3d to %s\n\n" "$B3D_ROOT"
  gh repo clone probcomp/b3d

  cd "$B3D_ROOT" || exit 1

  printf "\n→ checking out %s branch\n\n" "$B3D_BRANCH"
  git checkout "$B3D_BRANCH" origin/"$B3D_BRANCH"

  printf "\n→ installing pre-commit hooks %s\n\n" "$BIN_DIR"
  pre-commit install

  printf "\n→ installing environments %s\n\n" "$PWD/.pixi/envs"

  # temporary workaround for carvekit issue...
  local success=1
  local tries=5
  local attempt=0

  # solve default environment
  while [[ $success -ne 0 ]] && [[ $attempt -lt $tries ]]; do
    if pixi install -e default --locked; then
      success=0
      printf "\n\n→ success resolving 'pypi' requirements\n\n"
      printf "checking pytorch...\n\n"
      exec 3>&1 4>&2
      exec >/dev/null 2>&1
      if ! pixi run -e default python -c 'print("→ pypi resolved, checking pytorch...")' >/dev/null 2>&1; then
        pixi run -e default default -c 'print("→ retry: checking pytorch...")' >/dev/null 2>&1
      fi
      exec 1>&3 2>&4
    else
      attempt=$((attempt + 1))
      printf "\n→ resolving 'pypi' requirements (attempt %s of %s)...\n\n" "$attempt" "$tries"
    fi
  done

  # solve gpu environment
  tries=5
  attempt=0
  if nvidia-smi 2>/dev/null; then
    success=1
    while [[ $success -ne 0 ]] && [[ $attempt -lt $tries ]]; do
      printf "\n→ switching to GPU...\n\n"
      if pixi install -e gpu --locked; then
        success=0
        printf "\n→ success resolving 'pypi' requirements\n\n"
        printf "checking pytorch...\n"
        exec 3>&1 4>&2
        exec >/dev/null 2>&1
        if ! pixi run -e gpu python -c 'print("→ pypi resolved, checking pytorch...")' >/dev/null 2>&1; then
          pixi run -e gpu python -c 'print("→ retry: checking pytorch...")' >/dev/null 2>&1
        fi
        exec 1>&3 2>&4
      else
        attempt=$((attempt + 1))
        printf "\n→ resolving 'pypi' requirements (attempt %s of %s)...\n\n" "$attempt" "$tries"
      fi
    done
  fi

  if [[ $success -ne 0 ]]; then
    printf "\n→ unable to resolve 'pypi' requirements :(\n"
    printf "\n  run 'cd b3d && pixi clean cache && cd ../ && rm -rf b3d' and try again\n\n"
  else
    printf "\n✓ all environments ready\n\n"

    if [[ $B3D_REMOTE_MACHINE -eq 0 ]] && [[ $GCLOUD_AUTH_REMOTE -eq 1 ]]; then
      printf "✓ almost done!\n"
      printf "  1. restart your terminal or run: source ~/.bashrc\n"
      printf "  2. authenticate to gcloud by running: gcloud auth login --update-adc --force\n\n"
    else
      printf "✓ done! remember to restart your terminal or run: source ~/.bashrc\n\n"
    fi
  fi
}
__wrap__
