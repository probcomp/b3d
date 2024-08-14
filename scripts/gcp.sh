#!/usr/bin/env bash

if [[ $B3D_TEST_MODE -ne 1 ]]; then
  set -eo pipefail
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

COMMAND=""
GCP_VM="${GCP_VM:-}"
GCP_PROJECT="${GCP_PROJECT:-probcomp-caliban}"
GCP_REGION="${GCP_REGION:-us-west1}"
GCP_ZONE="${GCP_ZONE:-us-west1-a}"
GCP_CONNECT="${GCP_CONNECT:-ssh}"
GCP_DEBUG="${GCP_DEBUG:-}"
REMOTE_FORWARD="RemoteForward 8812 127.0.0.1:8812"
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"
GCLOUD_CREDS_FILE="$HOME/.config/gcloud/application_default_credentials.json"
GCLOUD_CREDS_DIR="/home/$USER"

# Prints help
gcp-help() {
  cat <<-EOF

		Google Cloud tasks are configured using environment variables. To see
		current values, use 'pixi run gcp-env'.

		Name          Required   Description
		----------------------------------------------------------------------------
		GCP_VM        Yes        Name of your machine
		GCP_PROJECT   Yes        Name of Google Cloud Platform project
		GCP_REGION    Yes        Region (defaults to us-west1)
		GCP_ZONE      Yes        Zone (defaults to us-west1-a)
		SSH_CONFIG    Yes        Your SSH config file (defaults to ~/.ssh/config)
		GCP_DEBUG     No         Verbose SSH logging (defaults to off)

		You can set environment variables a few different ways.

		1. Add them to your shell configuration file (e.g., ~/.bashrc)

		2. Export them into the current shell session:

		   export GCP_VM=neyman-b3d-gpu
		   pixi run gcp-code

		3. Prepend to tasks, useful when working with multiple machines:

		   GCP_VM=aaron-experiment-1 pixi run gcp-start
		   GCP_VM=aaron-experiment-2 pixi run gcp-start
		   GCP_VM=aaron-experiment-3 pixi run gcp-start

	EOF
}

# Prints the environment.
gcp-env() {
  cat <<-EOF

		GCP_VM = ${GCP_VM:-}
		GCP_PROJECT = ${GCP_PROJECT:-}
		GCP_REGION = ${GCP_REGION:-}
		GCP_ZONE = ${GCP_ZONE:-}
		GCP_DEBUG="${GCP_DEBUG:-}"
		SSH_CONFIG = ${SSH_CONFIG:-}

	EOF
}

# Executes a gcloud command.
gcp-execute() {
  local command=("$@")
  "${command[@]}"
}

# Logs a user message.
gcp-log() {
  local msg="$1"
  printf "%s\n" "$msg"
  return 0
}

# Gets the active host name.
#
# Returns:
#   0 on success, echos host string
#   1 on error, GCP_VM required
#   2 on error, GCP_ZONE required
#   3 on error, GCP_PROJECT required
gcp-active-host() {
  if [ -z "$GCP_VM" ]; then
    echo "GCP_VM required"
    return 1
  fi
  if [ -z "$GCP_ZONE" ]; then
    echo "GCP_ZONE required"
    return 2
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 3
  fi

  local host="$GCP_VM.$GCP_ZONE.$GCP_PROJECT"
  echo "$host"
  return 0
}

# Gets the address name for GCP_VM.
#
# Returns:
#   0 in success, echos address name
#   1 on error, GCP_VM required
gcp-address-name() {
  if [ -z "$GCP_VM" ]; then
    echo "GCP_VM required"
    exit 1
  fi

  local address="${GCP_VM}-address"
  echo "$address"
  return 0
}

# Creates an address with 'gcloud compute addresses create'.
#
# Returns:
#   0 on success, echos address
#   1 on error, GCP_REGION required
#   2 on error, GCP_PROJECT required
#   3 on error, unable to get address name
#   4 on error, gcloud unable to create address
gcp-create-address() {
  if [ -z "$GCP_REGION" ]; then
    echo "GCP_REGION required"
    return 1
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 2
  fi

  local address
  local command
  local output

  if ! address=$(gcp-address-name); then
    echo "error: unable to get address ($address)"
    return 3
  fi

  command=(
    gcloud compute addresses create "$address"
    --region="$GCP_REGION"
    --project="$GCP_PROJECT"
  )

  if ! output=$(gcp-execute "${command[@]}"); then
    echo "error: gcloud unable to create address"
    echo "$output"
    return 4
  fi

  echo "$address"
  return 0
}

# Deletes an existing address with 'gcloud compute addresses delete'.
#
# Returns:
#   0 on success, address was deleted
#   1 on error, GCP_REGION required
#   2 on error, GCP_PROJECT required
#   3 on error, gcloud unable to delete address
gcp-delete-address() {
  if [ -z "$GCP_REGION" ]; then
    echo "GCP_REGION required"
    return 1
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 2
  fi

  local address
  local command

  address=$(gcp-address-name)

  command=(
    gcloud compute addresses delete "$address"
    --region="$GCP_REGION"
    --project="$GCP_PROJECT"
  )

  if ! gcp-execute "${command[@]}"; then
    return 3
  fi

  return 0
}

# Gets static IP for existing address with 'gcloud compute addresses describe'.
#
# Returns:
#   0 on success, echos address
#   1 on error, GCP_REGION required
#   2 on error, GCP_PROJECT required
#   3 on error, unable to get address name
#   4 on error, gcloud unable to get static IP
gcp-get-static-ip() {
  if [ -z "$GCP_REGION" ]; then
    echo "GCP_REGION required"
    return 1
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 2
  fi

  local address
  local command
  local ip

  if ! address=$(gcp-address-name); then
    echo "error: unable to create address ($address)"
    return 3
  fi

  command=(
    gcloud compute addresses describe "$address"
    --format="get(address)"
    --region="$GCP_REGION"
    --project="$GCP_PROJECT"
  )

  if ! ip=$(gcp-execute "${command[@]}"); then
    echo "error: $ip"
    return 4
  fi

  echo "$ip"
  return 0
}

# Creates a VM with 'gcloud compute instances create'.
#
# Takes:
#   address (optional) -- name of existing address created by `gcp-create-address`
#
# Returns:
#   0 on success, VM created
#   1 on error, address required for vscode
#   2 on error, GCP_VM required
#   3 on error, GCP_REGION required
#   4 on error, GCP_PROJECT required
#   5 on error, GCP_ZONE required
#   6 on error, GCP_PROJECT required
#   7 on error, could not get host
#   8 on error, gcloud create
gcp-create() {
  if [ "$GCP_CONNECT" == "vscode" ] && [ -z "$1" ]; then
    echo "address required (for the static IP) when creating a VM for vscode"
    return 1
  fi
  if [ -z "$GCP_VM" ]; then
    echo "error: GCP_VM is required"
    exit 2
  fi
  if [ -z "$GCP_REGION" ]; then
    echo "GCP_REGION required"
    return 3
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 4
  fi
  if [ -z "$GCP_ZONE" ]; then
    echo "GCP_ZONE required"
    return 5
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 6
  fi

  local address="$1"
  local command
  local host
  local result

  if ! host=$(gcp-active-host); then
    echo "error: could not get host ($host)"
    return 7
  fi

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
    command+=(--address="$address")
  fi

  if ! result=$(gcp-execute "${command[@]}"); then
    echo "error: gcp-create ($result)"
    return 8
  fi

  return 0
}

# Deletes a VM with 'gcloud compute instances delete'.
#
# Returns:
#  0 on success, GCP_VM deleted
#  1 on error, GCP_VM required
#  2 on error, GCP_PROJECT required
#  3 on error, GCP_ZONE required
#  4 on error, couldn't get host
#  5 on error, gcloud delete failed
gcp-delete() {
  if [ -z "$GCP_VM" ]; then
    echo "error: GCP_VM is required"
    return 1
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 2
  fi
  if [ -z "$GCP_ZONE" ]; then
    echo "GCP_ZONE required"
    return 3
  fi

  local host
  local command
  local result

  if ! host=$(gcp-active-host); then
    echo "error: could not get host ($host)"
    return 4
  fi

  command=(
    gcloud compute instances delete "$GCP_VM"
    --project="$GCP_PROJECT"
    --zone="$GCP_ZONE"
  )

  if ! gcp-execute "${command[@]}"; then
    return 5
  fi

  return 0
}

# Stops a VM with 'gcloud compute instances stop'.
#
# Returns:
#  0 on success, VM stopped
#  1 on error, GCP_VM required
#  2 on error, GCP_PROJECT required
#  3 on error, GCP_ZONE required
#  4 on error, couldn't get host
#  5 on error, gcloud stop failed
gcp-stop() {
  if [ -z "$GCP_VM" ]; then
    echo "error: GCP_VM is required"
    return 1
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 2
  fi
  if [ -z "$GCP_ZONE" ]; then
    echo "GCP_ZONE required"
    return 3
  fi

  local host
  local command
  local result

  if ! host=$(gcp-active-host); then
    echo "error: could not get host ($host)"
    return 4
  fi

  command=(
    gcloud compute instances stop "$GCP_VM"
    --project="$GCP_PROJECT"
    --zone="$GCP_ZONE"
  )

  if ! result=$(gcp-execute "${command[@]}"); then
    echo "error: gcp-stop ($result)"
    return 5
  fi

  return 0
}

# Gets VM status with 'gcloud compute instances describe'.
#
# Returns:
#  0 on success, VM status
#  1 on error, GCP_VM required
#  2 on error, GCP_PROJECT required
#  3 on error, GCP_ZONE required
#  4 on error, couldn't get host
#  5 on error, gcloud status failed
gcp-status() {
  if [ -z "$GCP_VM" ]; then
    echo "error: GCP_VM is required"
    return 1
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 2
  fi
  if [ -z "$GCP_ZONE" ]; then
    echo "GCP_ZONE required"
    return 3
  fi

  local host
  local command
  local status

  if ! host=$(gcp-active-host); then
    echo "error: could not get host ($host)"
    return 4
  fi

  command=(
    gcloud compute instances describe "$GCP_VM"
    --zone="$GCP_ZONE"
    --project="$GCP_PROJECT"
    --format='get(status)'
  )

  if ! status=$(gcp-execute "${command[@]}" 2>/dev/null); then
    echo "DOES_NOT_EXIST"
  else
    echo "$status"
  fi

  return 0
}

# Starts a VM with 'gcloud compute instances start'.
#
# Returns:
#  0 on success, VM started
#  1 on error, GCP_VM required
#  2 on error, GCP_PROJECT required
#  3 on error, GCP_ZONE required
#  4 on error, couldn't get host
#  5 on error, gcloud start failed
gcp-start() {
  if [ -z "$GCP_VM" ]; then
    echo "error: GCP_VM is required"
    return 1
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 2
  fi
  if [ -z "$GCP_ZONE" ]; then
    echo "GCP_ZONE required"
    return 3
  fi

  local host
  local command
  local result

  if ! host=$(gcp-active-host); then
    echo "error: could not get host ($host)"
    return 4
  fi

  command=(
    gcloud compute instances start "$GCP_VM"
    --project="$GCP_PROJECT"
    --zone="$GCP_ZONE"
  )

  if ! result=$(gcp-execute "${command[@]}"); then
    echo "error: gcp-start ($result)"
    return 5
  fi

  return 0
}

# Runs 'gcloud compute config-ssh' for GCP_PROJECT
#
# Returns:
#  0 on success, config-ssh completed
#  1 on error, GCP_PROJECT required
#  2 on error, gcp-config-ssh failed
gcp-config-ssh() {
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 1
  fi

  local command
  local result

  command=(
    gcloud compute config-ssh
    --project="$GCP_PROJECT"
  )

  if ! result=$(gcp-execute "${command[@]}" >/dev/null); then
    echo "error: gcp-config-ssh ($result)"
    return 2
  fi

  return 0
}

# Lists all VMS with 'gcloud compute instances list'.
#
# Returns:
#  0 on success, echos list of VMs
#  1 on error, GCP_PROJECT reuired
#  2 on error, gcp-list failed
gcp-list() {
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 1
  fi

  local command
  local result

  command=(
    gcloud compute instances list
    --project="$GCP_PROJECT"
  )

  if ! gcp-execute "${command[@]}"; then
    echo "error: gcp-list"
    return 2
  fi

  return 0
}

# Tries to start a VM.
#
# Takes:
#  status -- VM status
#
# Returns:
#  0 on success, VM started
#  1 on error, couldn't get VM status
#  2 on error, couldn't start VM
#  3 on failure, VM already running
#  4 on failure, VM doesn't exist
#  5 on error, unknown VM status
gcp-try-to-start-vm() {
  local status

  if ! status=$(gcp-status); then
    echo "$status"
    return 1
  fi

  case $status in
  TERMINATED)
    if ! status=$(gcp-start); then
      echo "$status"
      return 2
    fi
    ;;
  RUNNING)
    echo "$status"
    return 3
    ;;
  DOES_NOT_EXIST)
    echo "$status"
    return 4
    ;;
  *)
    echo "error: unknown status $status"
    return 5
    ;;
  esac

  echo "$status"
  return 0
}

# Adds RemoteForward details to ssh config file.
gcp-update-ssh-config-remote-forward() {
  local host=""
  local os=""
  local temp_file

  host=$(gcp-active-host)
  os=$(uname -s)

  gcp-log "→ setting up $host on $os with remote forwarding in $SSH_CONFIG"

  case $os in
  Darwin)
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

# Uses SCP to transfer gcloud credentials to GCP.
gcp-scp() {
  local command
  local src="$1"
  local dest="$2"

  echo "scp $src $dest"

  if ! [[ -e $src ]]; then
    echo "error: src not found $src"
    return 1
  fi

  if [[ -z ${dest} ]]; then
    echo "error: src not found $src"
    return 3
  fi

  if [ -z "$GCP_VM" ]; then
    echo "error: GCP_VM is required"
    return 3
  fi
  if [ -z "$GCP_PROJECT" ]; then
    echo "GCP_PROJECT required"
    return 4
  fi

  command=(
    gcloud compute scp
    "$src"
    "$USER@$GCP_VM:$dest/"
    --zone="$GCP_ZONE"
    --project="$GCP_PROJECT"
  )

  if ! gcp-execute "${command[@]}"; then
    echo "error: could not tranfer $src"
    return 5
  else
    return 0
  fi

}

# Polls VM on status with linear backoff until it's RUNNING.
gcp-wait-until-running() {
  local retry_count=5
  local wait_time=3
  local attempt=0
  local status

  gcp-log "→ checking VM status..."

  while [ $attempt -lt $retry_count ]; do
    if status=$(gcp-status); then
      if [ "$status" == "RUNNING" ]; then
        gcp-log "→ VM status $status"
        return 0
      else
        attempt=$((attempt + 1))
        echo "VM is '$status' (attempt $attempt, retry in $wait_time seconds)"
        sleep $wait_time
        wait_time=$((wait_time + 1))
      fi
    else
      echo "unable to get status: $status"
      return 1
    fi
  done

  echo "VM is not running yet, try again"
  return 1

}

# Polls VM on status with linear backoff until it's RUNNING.
gcp-wait-until-connected() {
  local retry_count=5
  local wait_time=3
  local attempt=0
  local status

  gcp-log "→ checking VM status..."

  while [ $attempt -lt $retry_count ]; do
    if ! gcp-execute "${command[@]}"; then
      attempt=$((attempt + 1))
      echo "ssh attempt $attempt, retry in $wait_time seconds..."
      sleep $wait_time
      wait_time=$((wait_time + 1))
    fi
  done

  echo ""
  return 1

}

# Connect to VM through SSH.
gcp-ssh() {
  local host
  local command

  local retry_count=5
  local wait_time=3
  local attempts=0

  host=$(gcp-active-host)
  command=(
    ssh
  )

  if [ -n "$GCP_DEBUG" ]; then
    command+=("-v")
  fi

  command+=(
    -o
    StrictHostKeyChecking=ask
    "$host"
  )

  gcp-log "→ ssh $host"

  if gcp-wait-until-running; then
    # scp gcloud creds
    # while [ $attempts -lt $retry_count ]; do
    #   if ! gcp-scp "$GCLOUD_CREDS_FILE" "$GCLOUD_CREDS_DIR"; then
    #     attempt=$((attempt + 1))
    #     echo "attempt $attempts, retry in $wait_time seconds..."
    #     sleep $wait_time
    #     wait_time=$((wait_time + 1))
    #   else
    #     break
    #   fi
    # done

    # if [[ $attempts -eq $retry_count ]]; then
    # 	echo "error: unable to transfer gloud creds"
    # 	return 2
    # else
    # 	attempt=0
    # fi

    # scp gh creds
    while [ $attempts -lt $retry_count ]; do
      if ! gcp-execute "${command[@]}"; then
        attempt=$((attempt + 1))
        echo "attempt $attempts, retry in $wait_time seconds..."
        sleep $wait_time
        wait_time=$((wait_time + 1))
      else
        return 0
      fi
    done
  fi

  echo "error: $host not running yet, try again"
  return 3
}

# Connect to VM through vscode using supplied static IP.
# Returns:
#  0 on success, launched vscode
#  1 on error, code command not found
#  2 on error, VM not running after retries
gcp-vscode() {
  local command
  local host
  local status

  host=$(gcp-active-host)
  os=$(uname -s)
  command=(
    code
    --folder-uri
    "vscode-remote://ssh-remote+$host/home/$USER"
  )

  if ! hash code 2>/dev/null; then
    local os
    echo "error: 'code' command was not found"
    if [ "$os" == "Darwin" ]; then
      echo "hint: from the vscode command palette, run"
      echo "  Shell Command: Install 'code' command in PATH"
    fi
    exit 1
  fi

  if status=$(gcp-wait-until-running); then
    gcp-execute "${command[@]}"
    return 0
  else
    echo "$status"
    return 2
  fi
}

# Dispatch to conncting through vscode or terminal.
gcp-connect() {
  if [ "$GCP_CONNECT" != "ssh" ] && [ "$GCP_CONNECT" != "vscode" ]; then
    echo "error: GCP_CONNECT can only be 'ssh' or 'vscode'"
    exit 1
  fi

  local host
  local status
  local address

  host=$(gcp-active-host)
  status=$(gcp-status)

  gcp-log "→ connecting to $host through $GCP_CONNECT"

  while [[ $status != "READY" ]]; do
    case $status in
    DOES_NOT_EXIST)
      gcp-log "→ vm does not exist, so a new vm will be created..."
      gcp-log "→ creating vm address"
      address=$(gcp-create-address)
      gcp-log "→ creating vm $host"
      gcp-create "$address"
      gcp-log "→ configuring vm ssh"
      gcp-config-ssh
      gcp-log "→ adding remote forwarding to $SSH_CONFIG..."
      gcp-update-ssh-config-remote-forward
      status=READY
      ;;
    TERMINATED)
      gcp-log "→ vm is stopped, starting it now..."
      gcp-start
      status=RUNNING
      ;;
    RUNNING)
      gcp-log "→ the vm is running..."
      gcp-config-ssh
      gcp-log "→ updating remote forwarding in $SSH_CONFIG..."
      gcp-update-ssh-config-remote-forward
      status=READY
      ;;
    *)
      echo "vm status is unknown: $status"
      exit 1
      ;;
    esac
  done

  case $GCP_CONNECT in
  vscode)
    gcp-log "→ connecting through vscode..."
    gcp-vscode
    return 0
    ;;
  ssh)
    gcp-log "→ connecting though terminal..."
    gcp-ssh
    return 0
    ;;
  esac
}

execute() {
  case "$1" in
  :gcp-help)
    gcp-help
    ;;
  :gcp-scp)
    gcp-scp
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
    gcp-update-ssh-config-remote-forward
    ;;
  :gcp-config-ssh)
    gcp-config-ssh
    ;;
  :gcp-delete)
    gcp-delete
    gcp-delete-address
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

  local result

  if execute "$@"; then
    exit 0
  else
    echo "$result"
    exit 1
  fi
}

if [[ $B3D_TEST_MODE -eq 1 ]]; then
  echo "entering test mode..."
else
  main "$@"
fi
