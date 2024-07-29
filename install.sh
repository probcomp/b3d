#!/usr/bin/env bash

set -euo pipefail

__wrap__() {

	PIXI_BIN="$HOME/.pixi/bin"
	PIPX_BIN="$HOME/.local/bin"
	export PATH=$PIXI_BIN:$PIPX_BIN:$PATH

	confirm_installation() {
		cat <<-EOF
			The purpose of this script is to install the 'b3d' environment using 'pixi'.

			→ Developer tools get globally installed into their own environment:
			  Installation prefix: "$HOME/.pixi"
			  Installed packages: 'pixi', 'keyring', 'gcloud', 'rerun-sdk', 'pre-commit'

			→ Python and system-level dependencies get installed locally into project environments:
			  Installation prefix: $PWD/.pixi/envs"
			  Installed pacakges: Run 'pixi info' to see dependencies listed by project environment.

		EOF

		while true; do
			read -p "Continue? (yes/no): " choice
			case "$choice" in
			yes | y)
				echo "Continuing with the installation..."
				break
				;;
			no | n)
				echo ""
				exit 1
				;;
			*)
				echo "Answer yes or no"
				;;
			esac
		done
	}

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
		if wget -qO- https://pixi.sh/install.sh | bash; then
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
		if command -v gcloud &>/dev/null; then
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
		elif [[ "$ACTIVE_ACCOUNT" == *"@developer.gserviceaccount.com" ]]; then
			return 1
		else
			return 0
		fi
	}

	rerun-installed() {
		if command -v rerun &>/dev/null; then
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
		if command -v "$PIXI_BIN/pre-commit" &>/dev/null; then
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

	update-shell-config-path() {
		local config="$1"
		local path="$2"
		if grep -n "$path" "$config" &>/dev/null; then
			return 0
		else
			echo "$path" >>"$config"
		fi
	}

	direnv-update-config() {
		local config="$1"
		local update='eval "$(direnv hook bash)"'
		if grep -n "$update" "$config" &>/dev/null; then
			return 0
		else
			echo "$update" >>"$config"
		fi
	}

	init-dev-environment() {
		local shell=""
		local shell_config=""
		local v=""
		local p=""
		local platform=""

		platform=$(uname -sm)
		confirm_installation
		printf "installing b3d environment on platform %s\n\n" "$platform"

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

		# keyring setup
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

		# gcloud setup
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

		# gcloud auth
		if ! check_gcloud_auth_login; then
			gcloud auth login --update-adc
		fi
		printf "→ you are authenticated to google artifact registry \o/\n\n"

		# install rerun-sdk
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

		# install pre-commit
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

		# update shell config
		echo "updating shell config path..."
		if ! update-shell-config-path \
			"$shell_config" \
			"export PATH=$PIXI_BIN:$PIPX_BIN:$PATH"; then
			echo "failed to update shell config"
			exit 1
		fi
		printf "  ✓ shell config path updated\n\n"

		# install environment
		echo "installing b3d project dependencies..."
		echo "  pixi install --locked --environment gpu..."
		pixi install --locked --environment gpu
		pixi info

		printf "\ninstall done! run these commands:\n"
		printf "  1) source %s\n" "$shell_config"
		printf "  3) pixi task list\n\n"

	}

	init-dev-environment
}
__wrap__
