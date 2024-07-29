#!/usr/bin/env bash

__wrap__() {

  # ............................................................................
  # b3d

  if hash conda 2>/dev/null; then
    conda init >/dev/null 2>&1
    conda deactivate >/dev/null 2>&1
    conda config --set auto_activate_base false
  fi

  B3D_CLONE=${B3D_CLONE:-}
  B3D_HOME=${B3D_HOME:-"$PWD"}
  B3D_BRANCH=${B3D_BRANCH:-main}
  B3D_CLONE_METHOD=${B3D_CLONE_METHOD:-"HTTPS"}
  B3D_REPO_SSH="git@github.com:probcomp/b3d.git"
  B3D_REPO_HTTPS="https://github.com/probcomp/b3d.git"

  # ............................................................................
  # pixi

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

  cat <<-EOF

		bootstrapping b3d install...
	EOF

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

  printf "\n→ updating shell config\n"

  case "$(basename "$SHELL")" in
  bash)
    # Default to bashrc as that is used in non login shells instead of the profile.
    LINE='eval "$(pixi completion --shell bash)"'
    update_shell ~/.bashrc "$LINE"
    ;;

  fish)
    LINE='pixi completion --shell fish | source'
    update_shell ~/.config/fish/config.fish "$LINE"
    ;;

  zsh)
    LINE="autoload -Uz compinit"
    update_shell ~/.zshrc "$LINE"
    LINE="compinit"
    update_shell ~/.zshrc "$LINE"
    LINE='eval "$(pixi completion --shell zsh)"'
    update_shell ~/.zshrc "$LINE"
    ;;

  *)
    echo "could not add pixi autocomplete to shell: $(basename "$SHELL")"
    echo "please see: https://pixi.sh/latest/#autocompletion"
    ;;
  esac

  PATH=$BIN_DIR:$PATH

  printf "\n→ installing pipx into %s\n\n" "$BIN_DIR"

  pixi global install pipx
  pipx ensurepath &>/dev/null

  if ! hash git 2>/dev/null; then
    printf "\n→ installing git into %s\n\n" "$BIN_DIR"
    pixi global install git
  fi

  B3D_HOME="$PWD"
  B3D_REPO="${B3D_HOME}/b3d"

  if [ -d "$B3D_REPO" ] || [ -n "$B3D_CLONE" ]; then
    cat <<-EOF

			✓ you can now install b3d

			cd b3d
			source ~/.bashrc
			pixi run b3d-install

		EOF
    exit 0
  else
    if ! mkdir "$B3D_REPO"; then
      echo "error: failed to create $B3D_REPO (check permissions)" >&2
      exit 1
    fi
    cd "$B3D_REPO" || exit 1
    if [ "$B3D_CLONE_METHOD" = "SSH" ]; then
      printf "\n→ cloning %s into current rectory\n\n" "$B3D_REPO_SSH"
      git clone "$B3D_REPO_SSH" .
    else
      printf "\n→ cloning %s into current directory\n\n" "$B3D_REPO_HTTPS"
      git clone "$B3D_REPO_HTTPS" .
    fi
    printf "\n→ checking out %s branch\n\n" "$B3D_BRANCH"
    git checkout -b "$B3D_BRANCH" origin/"$B3D_BRANCH"
  fi

  cat <<-EOF

		✓ you can now install b3d

		cd b3d
		source ~/.bashrc
		pixi run b3d-install

	EOF
}
__wrap__
