#!/usr/bin/env bash

set -euo pipefail

PIXI_BIN="$HOME/.pixi/bin"
PIPX_BIN="$HOME/.local/bin"
export PATH=$PIXI_BIN:$PIPX_BIN:$PATH

pipx-installed() {
	if command -v pipx &>/dev/null; then
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
	if command -v keyring &>/dev/null; then
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

gcloud-authenticated() {
	if gcloud auth application-default print-access-token >/dev/null; then
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
		return 1
	else
		return 0
	fi
}

main() {
	local v=""
	local p=""

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
}

main
