#!/usr/bin/env bash

# Test coverage for .#test-gcp.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

suite() {
  suite_addTest test-gcp-update-ssh-config-remote-forward
  suite_addTest test-gcp-create
  suite_addTest test-gcp-delete
  suite_addTest test-gcp-status
  suite_addTest test-gcp-start
  suite_addTest test-gcp-stop
  suite_addTest test-gcp-try-to-start-vm
  suite_addTest test-gcp-get-static-ip
  suite_addTest test-gcp-create-address
  suite_addTest test-gcp-delete-address
  suite_addTest test-gcp-log
  suite_addTest test-gcp-active-host
}

oneTimeSetUp() {
  export B3D_TEST_MODE=1
  source "$DIR/gcp.sh"
}

oneTimeTearDown() {
  export B3D_TEST_MODE=0
}

setUp() {
  if declare -f mock-gcp-execute >/dev/null; then
    declare -f gcp-execute >/tmp/original_gcp_execute
    gcp-execute() {
      mock-gcp-execute "$@"
    }
  fi
  GCP_VM="${GCP_VM:-}"
  GCP_PROJECT="${PROJECT_ID:-probcomp-caliban}"
  GCP_REGION="${GCP_REGION:-us-west1}"
  GCP_ZONE="${ZONE:-us-west1-a}"
  GCP_CONNECT="${GCP_CONNECT:-ssh}"
  REMOTE_FORWARD="RemoteForward 8812 127.0.0.1:8812"
  SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"
}

tearDown() {
  if [ -f /tmp/original_gcp_execute ]; then
    eval "$(cat /tmp/original_gcp_execute)"
    rm /tmp/original_gcp_execute
  fi
  unset -f mock-gcp-execute
}

debug() {
  echo "debug"
}

create_temp_ssh_config() {
  local temp_file

  temp_file=$(mktemp)

  cat <<EOF >"$temp_file"
Host georgeb3dtest1.us-west1-a.probcomp-caliban
    HostName 34.83.34.231
    IdentityFile /home/eighty/.ssh/google_compute_engine
    UserKnownHostsFile=/home/eighty/.ssh/google_compute_known_hosts
    HostKeyAlias=compute.5878676690657768597
    IdentitiesOnly=yes
    CheckHostIP=no

Host aaron-b3d-test.us-west1-a.probcomp-caliban
    RemoteForward 8812 127.0.0.1:8812
    HostName 34.145.107.172
    IdentityFile /home/eighty/.ssh/google_compute_engine
    UserKnownHostsFile=/home/eighty/.ssh/google_compute_known_hosts
    HostKeyAlias=compute.5238194004790400266
    IdentitiesOnly=yes
    CheckHostIP=no

Host gmatheos1.us-west1-a.probcomp-caliban
    HostName 35.230.14.123
    IdentityFile /home/eighty/.ssh/google_compute_engine
    UserKnownHostsFile=/home/eighty/.ssh/google_compute_known_hosts
    HostKeyAlias=compute.6292476908223395138
    IdentitiesOnly=yes
    CheckHostIP=no

Host sam-b3d-l4.us-west1-a.probcomp-caliban
    HostName 35.197.104.210
    IdentityFile /home/eighty/.ssh/google_compute_engine
    UserKnownHostsFile=/home/eighty/.ssh/google_compute_known_hosts
    HostKeyAlias=compute.7495268330374633569
    IdentitiesOnly=yes
    CheckHostIP=no
EOF

  echo $temp_file
}

test-gcp-update-ssh-config-remote-forward() {
  local ssh_config=$(
    cat <<'EOF'
Host aaron-b3d-test.us-west1-a.probcomp-caliban
    RemoteForward 8812 127.0.0.1:8812
    HostName 34.145.107.172
    IdentityFile /home/eighty/.ssh/google_compute_engine
    UserKnownHostsFile=/home/eighty/.ssh/google_compute_known_hosts
    HostKeyAlias=compute.5238194004790400266
    IdentitiesOnly=yes
    CheckHostIP=no

Host gmatheos1.us-west1-a.probcomp-caliban
    HostName 35.230.14.123
    IdentityFile /home/eighty/.ssh/google_compute_engine
    UserKnownHostsFile=/home/eighty/.ssh/google_compute_known_hosts
    HostKeyAlias=compute.6292476908223395138
    IdentitiesOnly=yes
    CheckHostIP=no

Host sam-b3d-l4.us-west1-a.probcomp-caliban
    HostName 35.197.104.210
    IdentityFile /home/eighty/.ssh/google_compute_engine
    UserKnownHostsFile=/home/eighty/.ssh/google_compute_known_hosts
    HostKeyAlias=compute.7495268330374633569
    IdentitiesOnly=yes
    CheckHostIP=no
EOF
  )

  local ssh_config_updated=$(
    cat <<'EOF'
Host aaron-b3d-test.us-west1-a.probcomp-caliban
    RemoteForward 8812 127.0.0.1:8812
    HostName 34.145.107.172
    IdentityFile /home/eighty/.ssh/google_compute_engine
    UserKnownHostsFile=/home/eighty/.ssh/google_compute_known_hosts
    HostKeyAlias=compute.5238194004790400266
    IdentitiesOnly=yes
    CheckHostIP=no

Host gmatheos1.us-west1-a.probcomp-caliban
    HostName 35.230.14.123
    IdentityFile /home/eighty/.ssh/google_compute_engine
    UserKnownHostsFile=/home/eighty/.ssh/google_compute_known_hosts
    HostKeyAlias=compute.6292476908223395138
    IdentitiesOnly=yes
    CheckHostIP=no

Host sam-b3d-l4.us-west1-a.probcomp-caliban
    RemoteForward 8812 127.0.0.1:8812
    HostName 35.197.104.210
    IdentityFile /home/eighty/.ssh/google_compute_engine
    UserKnownHostsFile=/home/eighty/.ssh/google_compute_known_hosts
    HostKeyAlias=compute.7495268330374633569
    IdentitiesOnly=yes
    CheckHostIP=no
EOF
  )

  OS="Darwin"
  GCP_VM=""
  local host_with="aaron-b3d-test"
  local host_without="sam-b3d-l4"
  local expected
  local actual
  local this_machine=$(uname -s)

  # mock uname
  uname() {
    echo "$OS"
  }

  # test adding remote forward to Darwin
  ssh_config_temp=$(mktemp)
  echo "$ssh_config" >"$ssh_config_temp"
  ssh_config_updated_temp=$(mktemp)
  echo "$ssh_config_updated" >"$ssh_config_updated_temp"
  SSH_CONFIG="$ssh_config_temp"
  GCP_VM="$host_without"
  OS="Darwin"
  gcp-update-ssh-config-remote-forward >/dev/null
  status=$?
  expected="$ssh_config_updated"
  actual=$(cat "$ssh_config_temp")
  $_ASSERT_TRUE_ $status
  $_ASSERT_EQUALS_ '"$expected"' '"$actual"'

  # Linux can test both Darwin and Linux, but Darwin cannot
  if [[ $this_machine != "Darwin" ]]; then
    # test adding remote forward to Linux
    ssh_config_temp=$(mktemp)
    SSH_CONFIG="$ssh_config_temp"
    echo "$ssh_config" >"$ssh_config_temp"
    GCP_VM="$host_without"
    OS="Linux"
    gcp-update-ssh-config-remote-forward >/dev/null
    status=$?
    expected="$ssh_config_updated"
    actual=$(cat "$ssh_config_temp")
    $_ASSERT_TRUE_ $status
    $_ASSERT_EQUALS_ '"$expected"' '"$actual"'
  fi
  # restore uname
  unset -f uname
}

test-gcp-get-static-ip() {
  local result
  local status

  GCP_VM="name"
  GCP_ZONE="zone"
  GCP_PROJECT="project"
  GCP_IP="123.45.67.890"

  unset -f mock-gcp-execute
  mock-gcp-execute() {
    $_ASSERT_EQUALS_ '"gcloud"' '"$1"'
    $_ASSERT_EQUALS_ '"compute"' '"$2"'
    $_ASSERT_EQUALS_ '"addresses"' '"$3"'
    $_ASSERT_EQUALS_ '"describe"' '"$4"'
    $_ASSERT_EQUALS_ '"${GCP_VM}-address"' '"$5"'
    $_ASSERT_EQUALS_ '"--format=get(address)"' '"$6"'
    $_ASSERT_EQUALS_ '"--region=$GCP_REGION"' '"$7"'
    $_ASSERT_EQUALS_ '"--project=$GCP_PROJECT"' '"$8"'
    echo "$GCP_IP"
    return 0
  }

  gcp-execute() {
    mock-gcp-execute "$@"
  }

  result=$(gcp-get-static-ip)
  status=$?
  $_ASSERT_TRUE_ $status
  $_ASSERT_EQUALS_ '"$GCP_IP"' '"$result"'
}

test-gcp-create-address() {
  local address
  local status

  GCP_VM="name"
  GCP_ZONE="zone"
  GCP_PROJECT="project"

  unset -f mock-gcp-execute
  mock-gcp-execute() {
    $_ASSERT_EQUALS_ '"gcloud"' '"$1"'
    $_ASSERT_EQUALS_ '"compute"' '"$2"'
    $_ASSERT_EQUALS_ '"addresses"' '"$3"'
    $_ASSERT_EQUALS_ '"create"' '"$4"'
    $_ASSERT_EQUALS_ '"${GCP_VM}-address"' '"$5"'
    $_ASSERT_EQUALS_ '"--region=$GCP_REGION"' '"$6"'
    $_ASSERT_EQUALS_ '"--project=$GCP_PROJECT"' '"$7"'
    return 0
  }

  gcp-execute() {
    mock-gcp-execute "$@"
  }

  address=$(gcp-create-address)
  status=$?
  $_ASSERT_TRUE_ $status
  $_ASSERT_EQUALS_ '"$GCP_VM-address"' '"$address"'
}

test-gcp-delete-address() {
  GCP_VM="name"
  GCP_ZONE="zone"
  GCP_PROJECT="project"
  GCP_ADDRESS="$GCP_VM-address"
  local status
  local address

  unset -f mock-gcp-execute
  mock-gcp-execute() {
    $_ASSERT_EQUALS_ '"gcloud"' '"$1"'
    $_ASSERT_EQUALS_ '"compute"' '"$2"'
    $_ASSERT_EQUALS_ '"addresses"' '"$3"'
    $_ASSERT_EQUALS_ '"delete"' '"$4"'
    $_ASSERT_EQUALS_ '"$GCP_ADDRESS"' '"$5"'
    $_ASSERT_EQUALS_ '"--region=$GCP_REGION"' '"$6"'
    $_ASSERT_EQUALS_ '"--project=$GCP_PROJECT"' '"$7"'
    return 0
  }

  gcp-execute() {
    mock-gcp-execute "$@"
  }

  address=$(gcp-delete-address "$GCP_ADDRESS")
  status=$?
  $_ASSERT_TRUE_ $status
}

test-gcp-create() {
  GCP_VM="name"
  GCP_ADDRESS="$GCP_VM-address"
  GCP_CONNECT="vscode"
  local status

  unset -f mock-gcp-execute
  mock-gcp-execute() {
    $_ASSERT_EQUALS_ '"gcloud"' '"$1"'
    $_ASSERT_EQUALS_ '"compute"' '"$2"'
    $_ASSERT_EQUALS_ '"instances"' '"$3"'
    $_ASSERT_EQUALS_ '"create"' '"$4"'
    $_ASSERT_EQUALS_ '"$GCP_VM"' '"$5"'
    $_ASSERT_EQUALS_ '"--project=$GCP_PROJECT"' '"$6"'
    $_ASSERT_EQUALS_ '"--region=$GCP_ZONE"' '"$7"'
    return 0
  }

  gcp-execute() {
    mock-gcp-execute "$@"
  }

  gcp-create "$GCP_ADDRESS" >/dev/null
  status=$?
  $_ASSERT_TRUE_ $status

  gcp-create >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status

  GCP_CONNECT="vscode"
  gcp-create >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
}

test-gcp-delete() {
  GCP_VM="name"
  local status

  unset -f mock-gcp-execute
  mock-gcp-execute() {
    $_ASSERT_EQUALS_ '"gcloud"' '"$1"'
    $_ASSERT_EQUALS_ '"compute"' '"$2"'
    $_ASSERT_EQUALS_ '"instances"' '"$3"'
    $_ASSERT_EQUALS_ '"delete"' '"$4"'
    $_ASSERT_EQUALS_ '"$GCP_VM"' '"$5"'
    $_ASSERT_EQUALS_ '"--project=$GCP_PROJECT"' '"$6"'
    $_ASSERT_EQUALS_ '"--zone=$GCP_ZONE"' '"$7"'
    return 0
  }

  gcp-execute() {
    mock-gcp-execute "$@"
  }

  gcp-delete >/dev/null
  status=$?
  $_ASSERT_TRUE_ $status

  unset GCP_VM
  gcp-delete >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
}

test-gcp-stop() {
  GCP_VM="name"
  local status

  unset -f mock-gcp-execute
  mock-gcp-execute() {
    $_ASSERT_EQUALS_ '"gcloud"' '"$1"'
    $_ASSERT_EQUALS_ '"compute"' '"$2"'
    $_ASSERT_EQUALS_ '"instances"' '"$3"'
    $_ASSERT_EQUALS_ '"stop"' '"$4"'
    $_ASSERT_EQUALS_ '"$GCP_VM"' '"$5"'
    $_ASSERT_EQUALS_ '"--project=$GCP_PROJECT"' '"$6"'
    $_ASSERT_EQUALS_ '"--zone=$GCP_ZONE"' '"$7"'
    return 0
  }

  gcp-execute() {
    mock-gcp-execute "$@"
  }

  gcp-stop >/dev/null
  status=$?
  $_ASSERT_TRUE_ $status

  unset GCP_VM
  gcp-stop >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
}

test-gcp-start() {
  GCP_VM="name"
  local status

  unset -f mock-gcp-execute
  mock-gcp-execute() {
    $_ASSERT_EQUALS_ '"gcloud"' '"$1"'
    $_ASSERT_EQUALS_ '"compute"' '"$2"'
    $_ASSERT_EQUALS_ '"instances"' '"$3"'
    $_ASSERT_EQUALS_ '"start"' '"$4"'
    $_ASSERT_EQUALS_ '"$GCP_VM"' '"$5"'
    $_ASSERT_EQUALS_ '"--project=$GCP_PROJECT"' '"$6"'
    $_ASSERT_EQUALS_ '"--zone=$GCP_ZONE"' '"$7"'
    return 0
  }

  gcp-execute() {
    mock-gcp-execute "$@"
  }

  gcp-start >/dev/null
  status=$?
  $_ASSERT_TRUE_ $status

  unset GCP_VM
  gcp-start >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
}

test-gcp-status() {
  GCP_VM="name"
  EXECUTE_STATUS=0
  local status
  local result

  unset -f mock-gcp-execute
  mock-gcp-execute() {
    $_ASSERT_EQUALS_ '"gcloud"' '"$1"'
    $_ASSERT_EQUALS_ '"compute"' '"$2"'
    $_ASSERT_EQUALS_ '"instances"' '"$3"'
    $_ASSERT_EQUALS_ '"describe"' '"$4"'
    $_ASSERT_EQUALS_ '"$GCP_VM"' '"$5"'
    $_ASSERT_EQUALS_ '"--zone=$GCP_ZONE"' '"$6"'
    $_ASSERT_EQUALS_ '"--project=$GCP_PROJECT"' '"$7"'
    $_ASSERT_EQUALS_ '"--format=get(status)"' '"$8'
    return $EXECUTE_STATUS
  }

  gcp-execute() {
    mock-gcp-execute "$@"
  }

  gcp-status >/dev/null
  status=$?
  $_ASSERT_TRUE_ $status

  EXECUTE_STATUS=1
  result=$(gcp-status) >/dev/null
  status=$?
  $_ASSERT_TRUE_ $status
  $_ASSERT_EQUALS_ '"DOES_NOT_EXIST"' '"$result"'

  unset GCP_VM
  gcp-status >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
}

test-gcp-try-to-start-vm() {
  GCP_VM="name"

  STATUS_RESULT=""
  STATUS_RETURN=0
  START_RESULT=""
  START_RETURN=0

  local status
  local result

  unset -f mock-gcp-execute
  mock-gcp-execute() {
    local caller="$1"
    shift

    if [[ $caller == "gcp-status" ]]; then
      echo "$STATUS_RESULT"
      return $STATUS_RETURN
    elif [[ $caller == "gcp-start" ]]; then
      echo "$START_RESULT"
      return $START_RETURN
    fi
  }

  gcp-execute() {
    mock-gcp-execute "${FUNCNAME[1]}" "$@"
  }

  STATUS_RESULT="TERMINATED"
  STATUS_RETURN=0
  START_RESULT=""
  START_RETURN=0
  gcp-try-to-start-vm >/dev/null
  status=$?
  $_ASSERT_TRUE_ $status

  STATUS_RESULT="TERMINATED"
  STATUS_RETURN=0
  START_RESULT=""
  START_RETURN=1
  gcp-try-to-start-vm >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
  $_ASSERT_EQUALS_ $status 2

  STATUS_RESULT="RUNNING"
  STATUS_RETURN=0
  START_RESULT=""
  START_RETURN=0
  gcp-try-to-start-vm >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
  $_ASSERT_EQUALS_ $status 3

  STATUS_RESULT="DOES_NOT_EXIST"
  STATUS_RETURN=0
  START_RESULT=""
  START_RETURN=0
  gcp-try-to-start-vm >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
  $_ASSERT_EQUALS_ $status 4

  STATUS_RESULT="UNKNOWN_STATUS"
  STATUS_RETURN=0
  START_RESULT=""
  START_RETURN=0
  gcp-try-to-start-vm >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
  $_ASSERT_EQUALS_ $status 5
}

test-gcp-log() {
  local msg="hello, world"
  local log

  log=$(gcp-log "$msg") >/dev/null

  $_ASSERT_TRUE_ "$?"
  $_ASSERT_EQUALS_ '"$msg"' '"$log"'
}

test-gcp-active-host() {
  local host
  local status

  GCP_VM="name"
  GCP_ZONE="zone"
  GCP_PROJECT="project"
  host=$(gcp-active-host) >/dev/null
  status=$?
  $_ASSERT_TRUE_ $status
  $_ASSERT_EQUALS_ '"$GCP_VM.$GCP_ZONE.$GCP_PROJECT"' '"$host"'

  GCP_VM=""
  GCP_ZONE="zone"
  GCP_PROJECT="project"
  host=$(gcp-active-host) >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
  $_ASSERT_EQUALS_ '"1"' '"$status"'

  GCP_VM="name"
  GCP_ZONE=""
  GCP_PROJECT="project"
  host=$(gcp-active-host) >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
  $_ASSERT_EQUALS_ '"2"' '"$status"'

  GCP_VM="name"
  GCP_ZONE="zone"
  GCP_PROJECT=""
  host=$(gcp-active-host) >/dev/null
  status=$?
  $_ASSERT_FALSE_ $status
  $_ASSERT_EQUALS_ '"3"' '"$status"'
}

. ./shunit2.sh
