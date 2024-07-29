#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

COMMAND=""
GCP_VM="${GCP_VM:-}"
GCP_PROJECT="${PROJECT_ID:-probcomp-caliban}"
GCP_REGION="${GCP_REGION:-us-west1}"
GCP_ZONE="${ZONE:-us-west1-a}"
GCP_CONNECT="${GCP_CONNECT:-ssh}"
REMOTE_FORWARD="RemoteForward 8812 127.0.0.1:8812"
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"

gcp-help() {
  cat <<-EOF

		GCP tasks are configured using environment variables, listed here with current values:

		  GCP_VM = ${GCP_VM:-}
		  GCP_PROJECT = ${GCP_PROJECT:-}
		  GCP_CONNECT = ${GCP_CONNECT:-}
		  GCP_REGION = ${GCP_REGION:-}
		  GCP_ZONE = ${GCP_ZONE:-}
		  SSH_CONFIG = ${SSH_CONFIG:-}

		Many tasks require a VM name configured using the GCP_VM environment
		variable. Here's an example of connecting to a new VM through vscode:

		  export GCP_VM=neyman-b3d-gpu
		  pixi run gcp-code

	EOF
}

gcp-env() {
  cat <<-EOF

		GCP_VM = ${GCP_VM:-}
		GCP_PROJECT = ${GCP_PROJECT:-}
		GCP_CONNECT = ${GCP_CONNECT:-}
		GCP_REGION = ${GCP_REGION:-}
		GCP_ZONE = ${GCP_ZONE:-}
		SSH_CONFIG = ${SSH_CONFIG:-}

	EOF
}

gcp-log() {
  local msg="$1"
  printf "%s\n" "$msg"
  return 0
}

gcp-active-host() {
  local host="$GCP_VM.$GCP_ZONE.$GCP_PROJECT"
  echo "$host"
  return 0
}

gcp-address-name() {
  local address

  if [ -z "$GCP_VM" ]; then
    echo "error: GCP_VM is required"
    exit 1
  fi

  address="${GCP_VM}-address"
  echo "$address"
  return 0
}

gcp-create-address() {
  local address
  local command

  gcp-log "→ creating static ip address..."

  address=$(gcp-address-name)
  command=(
    gcloud compute addresses create "$address"
    --region="$GCP_REGION"
    --project="$GCP_PROJECT"
  )

  "${command[@]}"
  return 0
}

gcp-get-static-ip() {
  local address
  local command
  local ip

  address=$(gcp-address-name)
  command=(
    gcloud compute addresses describe "$address"
    --region="$GCP_REGION"
    --format="get(address)"
    --project="$GCP_PROJECT"
  )

  if ip=$("${command[@]}" >/dev/null 2>&1); then
    echo "$ip"
    return 0
  else
    return 1
  fi
}

gcp-delete-address() {
  local address
  local command

  gcp-log "→ deleting static ip address..."

  address=$(gcp-address-name)
  command=(
    gcloud compute addresses delete "$address"
    --region="$GCP_REGION"
    --project="$GCP_PROJECT"
  )

  if gcp-get-static-ip; then
    "${command[@]}"
    return 0
  fi
}

gcp-create() {
  local address
  local command
  local host

  if [ -z "$GCP_VM" ]; then
    echo "error: GCP_VM is required"
    exit 1
  fi

  host=$(gcp-active-host)
  gcp-log "→ creating $host"

  command=(
    gcloud compute instances create "$GCP_VM"
    --project="$GCP_PROJECT"
    --zone="$GCP_ZONE"
    --image-family="common-cu123-ubuntu-2204-py310"
    --image-project=deeplearning-platform-release
    --maintenance-policy=TERMINATE
    --boot-disk-size=400GB
    --machine-type=g2-standard-32
    --accelerator="type=nvidia-l4,count=1"
    --metadata="install-nvidia-driver=True"
  )

  if [ "$GCP_CONNECT" == "vscode" ]; then
    address=$(gcp-address-name)
    gcp-create-address
    echo "attaching static ip address..."
    command+=(--address="$address")
  fi

  "${command[@]}"
  return 0
}

gcp-delete() {
  local host
  local command

  host=$(gcp-active-host)
  gcp-log "→ deleting $host"

  command=(
    gcloud compute instances delete "$GCP_VM"
    --project="$GCP_PROJECT"
    --zone="$GCP_ZONE"
  )

  gcp-delete-address

  "${command[@]}"
  return 0
}

gcp-stop() {
  local host
  local command

  host=$(gcp-active-host)
  gcp-log "→ stopping $host"

  command=(
    gcloud compute instances stop "$GCP_VM"
    --project="$GCP_PROJECT"
    --zone="$GCP_ZONE"
  )

  "${command[@]}"
  return 0
}

gcp-status() {
  local status
  local command

  command=(
    gcloud compute instances describe "$GCP_VM"
    --zone="$GCP_ZONE"
    --project="$GCP_PROJECT"
    --format='get(status)'
  )

  if ! status=$("${command[@]}" 2>/dev/null); then
    echo "DOES_NOT_EXIST"
  else
    echo "$status"
  fi

}

gcp-start() {
  local status
  local host
  local command

  host=$(gcp-active-host)
  gcp-log "→ stopping $host"

  status=$(gcp-status)
  case $status in
  TERMINATED)
    command=(
      gcloud compute instances start "$GCP_VM"
      --project="$GCP_PROJECT"
      --zone="$GCP_ZONE"
    )
    "${command[@]}"
    return 0
    ;;
  RUNNING)
    gcp-log "✓ $GCP_VM already running"
    return 0
    ;;
  DOES_NOT_EXIST)
    echo "$GCP_VM does not exist"
    ;;
  *)
    echo "error: unknown status $status"
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
  local command

  gcp-log "→ updating $SSH_CONFIG"

  command=(
    gcloud compute config-ssh
    --project="$GCP_PROJECT"
  )

  "${command[@]}" >/dev/null
  return 0
}

gcp-ssh() {
  local host=""
  local retry_count=5
  local wait_time=3
  local attempt=0

  host=$(gcp-active-host)
  gcp-log "→ ssh $host"

  while [ $attempt -lt $retry_count ]; do
    if ssh -o StrictHostKeyChecking=ask "$host"; then
      return 0
    else
      echo "ssh attempt $((attempt + 1)), retrying in $wait_time seconds..."
      sleep $wait_time
      attempt=$((attempt + 1))
      wait_time=$((wait_time + 1))
    fi
  done

  echo "error: failed to shh after $retry_count attempts"
  return 1
}

gcp-list() {
  local command

  command=(
    gcloud compute instances list
    --project="$GCP_PROJECT"
  )

  "${command[@]}"
  return 0
}

gcp-vscode() {
  local static_ip="$1"
  local retry_count=5
  local wait_time=3
  local attempt=0
  local os
  local command
  local host

  host=$(gcp-active-host)
  os=$(uname -s)
  command=(
    code
    --folder-uri
    "vscode-remote://ssh-remote+$host/home/$USER"
  )

  if ! hash code 2>/dev/null; then
    echo "error: 'code' command was not found"
    if [ "$os" == "Darwin" ]; then
      echo "try vscode command palette:"
      echo "  Shell Command: Install 'code' command in PATH"
    fi
    exit 1
  fi

  while [ $attempt -lt $retry_count ]; do
    status=$(gcp-status)
    if [ "$status" == "RUNNING" ]; then
      gcp-log "→ connecting to vscode through $static_ip"
      "${command[@]}"
      return 0
    else
      echo "connection attempt $((attempt + 1)), retry in $wait_time sec..."
      sleep $wait_time
      attempt=$((attempt + 1))
      wait_time=$((wait_time + 1))
    fi
  done

  echo "error: failed to connect after $retry_count attempts"
  return 1
}

gcp-connect() {
  local host
  local status
  local static_ip

  if [ "$GCP_CONNECT" != "ssh" ] && [ "$GCP_CONNECT" != "vscode" ]; then
    echo "error: GCP_CONNECT can only be 'ssh' or 'vscode'"
    exit 1
  fi

  host=$(gcp-active-host)
  gcp-log "→ connecting to $host through $GCP_CONNECT"

  status=$(gcp-status)

  case $status in
  DOES_NOT_EXIST)
    gcp-log "→ vm does not exist, so a new vm will be created..."
    gcp-create
    gcp-config-ssh
    gcp-remote-forward
    ;;
  RUNNING)
    gcp-log "→ the vm is running..."
    ;;
  *)
    echo "error: vm status is unknown '$status'"
    exit 1
    ;;
  esac

  case $GCP_CONNECT in
  vscode)
    static_ip=$(gcp-get-static-ip)
    gcp-vscode "$static_ip"
    return 0
    ;;
  ssh)
    gcp-ssh
    return 0
    ;;
  esac
}

interpret() {
  case "$1" in
  :gcp-help)
    gcp-help
    ;;
  :gcp-env)
    gcp-env
    ;;
  :gcp-connect-vscode)
    GCP_CONNECT="vscode"
    gcp-connect
    ;;
  :gcp-connect-terminal)
    GCP_CONNECT="ssh"
    gcp-connect
    ;;
  :gcp-create)
    gcp-create
    ;;
  :gcp-static-ip)
    gcp-get-static-ip
    ;;
  :gcp-create-address)
    gcp-create-address
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
  :gcp-start)
    gcp-start
    ;;
  :gcp-stop)
    gcp-stop
    ;;
  :gcp-status)
    gcp-status
    ;;
  :gcp-ssh)
    gcp-ssh
    ;;
  *)
    echo "error: unknown command $COMMAND"
    exit 1
    ;;
  esac
}

main() {
  if ! cd "$PROJECT_ROOT"; then
    echo "failed to change into project root: $PROJECT_ROOT"
    exit 1
  fi
  interpret "$@"
}

main "$@"
