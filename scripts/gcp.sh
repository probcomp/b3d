#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

COMMAND=""
GCP_NAME="${GCP_NAME:-}"
PROJECT_ID="${PROJECT_ID:-probcomp-caliban}"
ZONE="${ZONE:-us-west1-a}"
REMOTE_FORWARD="RemoteForward 8812 127.0.0.1:8812"
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"

gcp-log() {
	local msg="$1"
	printf "%s\n" "$msg"
}

gcp-active-host() {
	local host="$GCP_NAME.$ZONE.$PROJECT_ID"

	if [ -z "$GCP_NAME" ]; then
		echo "no active host! set GCP_NAME like this → export GCP_NAME=your-instance-name (then try again)"
		return 1
	else
		echo "$host"
		return 0
	fi
}

gcp-create() {
	gcp-log "→ creating..."

	if ! gcp-active-host; then
		exit 1
	fi

	gcloud compute instances create "$GCP_NAME" \
		--project="$PROJECT_ID" \
		--zone="$ZONE" \
		--image-family="common-cu123-ubuntu-2204-py310" \
		--image-project=deeplearning-platform-release \
		--maintenance-policy=TERMINATE \
		--boot-disk-size=400GB \
		--machine-type g2-standard-32 \
		--accelerator="type=nvidia-l4,count=1" \
		--metadata="install-nvidia-driver=True"

}

gcp-delete() {
	gcp-log "→ deleting"
	gcloud \
		compute \
		instances \
		delete \
		"$GCP_NAME" \
		--project="$PROJECT_ID" \
		--zone="$ZONE"
}

gcp-bootstrap() {
	gcp-log "→ bootstrapping with scripts/bootstrap.sh"
	gcloud \
		compute \
		instances \
		add-metadata \
		"$GCP_NAME" \
		--zone="$ZONE" \
		--project="$PROJECT_ID" \
		--metadata-from-file startup-script=scripts/bootstrap.sh
}

gcp-stop() {
	gcp-log "→ stopping"
	gcloud \
		compute \
		instances \
		stop \
		"$GCP_NAME" \
		--project="$PROJECT_ID" \
		--zone="$ZONE"
}

gcp-status() {
	gcp-log "→ status"
	gcloud \
		compute \
		instances \
		describe \
		"$GCP_NAME" \
		--zone="$ZONE" \
		--project="$PROJECT_ID" \
		--format='get(status)'
}

gcp-start() {
	gcp-log "→ starting"
	local status=""

	status=$(gcp-status)
	case $status in
	TERMINATED)
		gcloud \
			compute \
			instances \
			start \
			"$GCP_NAME" \
			--project="$PROJECT_ID" \
			--zone="$ZONE"
		;;
	RUNNING)
		gcp-log "✓ $GCP_NAME already running"
		;;
	*)
		echo "unknown status $status"
		exit 1
		;;
	esac
}

gcp-remote-forward() {
	local host=""
	local os=""
	local temp_file

	host=$(gcp-active-host)
	os=$(uname -s)

	gcp-log "→ setting up remote forwarding in $SSH_CONFIG"

	case $os in
	Darwin)
		if grep -q "Host $host" "$SSH_CONFIG"; then
			if ! grep -q "$REMOTE_FORWARD" "$SSH_CONFIG"; then
				temp_file=$(mktemp)
				awk -v host="$host" -v rf="$REMOTE_FORWARD" '
                $0 ~ "Host " host {
                    print $0
                    print "    " rf
                    next
                }
                { print }
            ' "$SSH_CONFIG" >"$temp_file"
				chmod 600 "$temp_file"
				mv "$temp_file" "$SSH_CONFIG"
				gcp-log "✓ remote forwarding set: $SSH_CONFIG $host $REMOTE_FORWARD"
			fi
		else
			gcp-log "(error) $host not defined in $SSH_CONFIG"
			exit 1
		fi
		;;
	Linux)
		if grep -q "Host $host" "$SSH_CONFIG"; then
			gcp-log "✓ host $host found in SSH config"

			if awk -v host="$host" -v remote_forward="$REMOTE_FORWARD" '
            $0 ~ "Host " host { in_host_block = 1 }
            in_host_block && $0 ~ remote_forward { found = 1; exit }
            in_host_block && $0 ~ /^Host / && !($0 ~ "Host " host) { in_host_block = 0 }
            END { exit !found }
        ' "$SSH_CONFIG"; then
				gcp-log "✓ remote forwarding already set for $host"
			else
				sed -i "/Host $host/a\\    $REMOTE_FORWARD" "$SSH_CONFIG"
				gcp-log "✓ remote forwarding set: $SSH_CONFIG $host $REMOTE_FORWARD"
			fi
		else
			echo "$host is not defined in $SSH_CONFIG"
			exit 1
		fi
		;;
	*)
		echo "unknown os $os"
		exit 1
		;;
	esac
}

gcp-config-ssh() {
	gcp-log "→ updating $SSH_CONFIG"
	gcloud \
		compute \
		config-ssh \
		--project="$PROJECT_ID" \
		>/dev/null
}

gcp-ssh() {
	gcp-log "→ ssh"
	local host=""

	host=$(gcp-active-host)
	ssh -o StrictHostKeyChecking=ask "$host"
}

gcp-list() {
	gcloud \
		compute \
		instances \
		list \
		--project="$PROJECT_ID"
}

interpret() {
	local command="$1"
	case "$command" in
	:gcp-create)
		gcp-create
		;;
	:gcp-list)
		gcp-list
		;;
	:gcp-active-host)
		gcp-active-host
		;;
	:gcp-remote-forward)
		gcp-remote-forward
		;;
	:gcp-config-ssh)
		gcp-config-ssh
		;;
	:gcp-delete)
		gcp-delete
		;;
	:gcp-bootstrap)
		gcp-bootstrap
		;;
	:gcp-start)
		gcp-start
		;;
	:gcp-stop)
		gcp-stop
		;;
	:gcp-connect)
		gcp-connect
		;;
	:gcp-status)
		gcp-status
		;;
	:gcp-ssh)
		gcp-ssh
		;;
	*)
		echo "unknown command $COMMAND"
		exit 1
		;;
	esac
}

main() {
	if ! cd "$PROJECT_ROOT"; then
		echo "failed to change into the project root directory $PROJECT_ROOT"
		exit 1
	fi
	if [ -z "$GCP_NAME" ]; then
		echo "warning: GCP_NAME not set → rememver to: export GCP_NAME=your-instance-name"
		return 1
	else
		interpret "$@"
	fi
}

main "$@"
