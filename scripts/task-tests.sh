#!/usr/bin/env bash

set -euo pipefail

PYTORCH_CACHE="$HOME/.cache/torch_extensions/*"

confirm-torch-cache() {
	cat <<-EOF
		Delete cached pytorch extensions in $PYTORCH_CACHE before running tests?
	EOF

	while true; do
		read -p "(yes/no): " choice
		case "$choice" in
		yes | y)
			echo "deleting $PYTORCH_CACHE"
			rm -rf "$PYTORCH_CACHE"
			return 0
			;;
		no | n)
			return 0
			;;
		*)
			echo "Answer yes or no"
			;;
		esac
	done
}

main() {
	confirm-torch-cache
	echo "Running unit tests..."
	cd "$PIXI_PROJECT_ROOT"
	pytest tests
}

main
