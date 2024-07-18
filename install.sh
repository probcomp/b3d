#!/usr/bin/env bash

set -euo pipefail

__wrap__() {

	PIXI_BIN="$HOME/.pixi/bin"
	PIPX_BIN="$HOME/.local/bin"
	export PATH=$PIXI_BIN:$PIPX_BIN:$PATH

	pixi-installed() {
		if command -v pixi &>/dev/null; then
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
		if curl -fsSL https://pixi.sh/install.sh | bash; then
			return 0
		else
			return 1
		fi
	}

	pixi-enable-autocomplete() {
		local shell=$1
		local config=$2

		case $shell in
		bash)
			echo 'eval "$(pixi completion --shell bash)"' >>"$config"
			;;
		zsh)
			echo 'eval "$(pixi completion --shell zsh)"' >>"$config"
			;;
		fish)
			echo 'pixi completion --shell fish | source' >>"$config"
			;;
		elvish)
			echo 'eval (pixi completion --shell elvish | slurp)' >>"$config"
			;;
		*)
			echo "unknown shell: $shell"
			i
			;;
		esac
	}

	get-current-shell() {
		local path=""
		local name=""

		if [ -n "$SHELL" ]; then
			path=$(echo $SHELL)
			name=$(basename $path)
			echo $name
			return 0
		else
			return 1
		fi
	}

	get-shell-config() {
		local shell=$1
		case $shell in
		bash)
			echo "$HOME/.bashrc"
			;;
		zsh)
			echo "$HOME/.zshrc"
			;;
		fish)
			echo "$HOME/.config/fish/config.fish"
			;;
		elvish)
			echo "$HOME/.elvish/rc.elv"
			;;
		*)
			echo "unknown shell: $shell"
			;;
		esac
	}

	init-dev-environment() {
		local shell=""
		local shell_config=""
		local v=""
		local p=""

		printf "installing b3d development environment...\n\n"

		# check shell
		echo "checking shell..."
		if ! shell=$(get-current-shell); then
			echo "SHELL not set"
			exit 1
		else
			printf "  ✓ shell is %s \n" "$shell"
		fi
		shell_config=$(get-shell-config $shell)
		if [ -e "$shell_config" ]; then
			printf "  ✓ shell config: %s \n\n" "$shell_config"
		else
			echo "SHELL config not found"
			exit 1
		fi

		# install pixi
		echo "checking pixi..."
		if ! pixi-installed; then
			echo "  installing pixi..."
			if ! pixi-global-install; then
				echo "couldn't install pixi"
				exit 1
			else
				pixi-enable-autocomplete "$shell" "$shell_config"
			fi
		else
			pixi self-update
		fi
		v=$(pixi --version)
		p=$(which pixi)
		printf "  ✓ %s installed (%s)\n\n" "$v" "$p"

		# setup google artifact registry access
		echo "checking google artifact registry..."
		pushd scripts &>/dev/null
		./check-auth.sh
		popd &>/dev/null

    # if [ "${GITHUB_CI:-}" = "true" ]; then
    #     rm "$PIXI_BIN/pixi"
    #     return 0
    # fi

		# install environment
		echo "installing b3d project dependencies..."
		echo "  pixi install..."
    pixi install --locked -vv

    # copy in libEGL.so
		# cp assets/system/libEGL.so .pixi/envs/default/x86_64-conda-linux-gnu/sysroot/usr/lib64/

		printf "\ninstall done! run these commands:\n"
		printf "  1) source %s\n" "$shell_config"
		printf "  3) pixi task list\n\n"

	}

	init-dev-environment
}
__wrap__
