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

pipx-installed() {
	if command -v "$PIXI_BIN/pipx" &>/dev/null; then
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
	if command -v "$PIPX_BIN/keyring" &>/dev/null; then
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
	if command -v "$PIXI_BIN/gcloud" &>/dev/null; then
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

gcloud-authenticated() {
    if [ "${GITHUB_CI:-}" = "true" ]; then
    return 0
	elif gcloud auth application-default print-access-token >/dev/null; then
		return 0
	else
		return 1
	fi
}

gcloud-init() {
	if ! gcloud init; then
		return 1
	else
		return 0
	fi
}

gcloud-auth-adc() {
	if ! gcloud auth application-default login; then
		returny 1
	else
		return 0
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
    	# install pipx
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

	# install keyring
	echo "checking keyring..."
	if ! keyring-installed; then
		echo "installing keyring..."
		if ! pipx-install-keyring; then
			echo "pipx couldn't install keyring"
			exit 1
		fi
	fi
	p=$(which keyring)
	printf "  ✓ keyring installed (%s)\n\n" "$p"

	# inject gcloud auth backend
	echo "checking google-artifact-registry-auth keyring backend..."
	if ! pipx-google-artifact-registry-injected; then
		echo "injecting google-artifact-registry-auth backend..."
		if ! pipx-inject-google-artifact-registry; then
			echo "pipx couldn't inject google artifact registry keyring"
			exit 1
		fi
	fi
	printf "  ✓ google artifact registry backend injected\n\n"

	# install gcloud
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

	# check gcloud auth
	echo "checking gcloud auth..."
	if ! gcloud-authenticated; then
		echo "authenticating gcloud..."
		if ! gcloud-auth-adc; then
			echo "gcloud not authenticated"
			exit 1
		fi
	fi
	printf "  ✓ gcloud authenticated\n\n"

	printf "→ you are authenticated to google artifact registry \o/\n"

		# install environment
		echo "installing b3d project dependencies..."
		echo "  pixi install..."
    pixi install --locked
    pixi info

    # copy in libEGL.so
		#cp assets/system/libEGL.so .pixi/envs/default/x86_64-conda-linux-gnu/sysroot/usr/lib64/

		printf "\ninstall done! run these commands:\n"
		printf "  1) source %s\n" "$shell_config"
		printf "  3) pixi task list\n\n"

	}

	init-dev-environment
}
__wrap__
