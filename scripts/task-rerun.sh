#!/usr/bin/env bash

# This script launches rerun.

set -euo pipefail

confirm-launch() {
	cat <<-EOF
		Launch rerun for remote testing?
	EOF

	while true; do
		read -p "(yes/no): " choice
		case "$choice" in
		yes | y)
			echo "launching rerun..."
			return 0
			;;
		no | n)
			return 1
			;;
		*)
			echo "Answer yes or no"
			;;
		esac
	done
}

main() {
	cd "$PIXI_PROJECT_ROOT"
	if pgrep -l rerun >/dev/null; then
		echo "rerun is already running"
		exit 0
	else
		if confirm-launch; then
			rerun --port 8812 &
			sleep 2
			exit 0
		fi
	fi
}

main
