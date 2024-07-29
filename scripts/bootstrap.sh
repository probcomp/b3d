#!/usr/bin/env bash

set -euo pipefail

conda init --reverse --all
conda config --set auto_activate_base false
source "$HOME/.bashrc"

if [ -d "b3d" ]; then
	echo "b3d exists"
else
	echo "clone b3d..."
	git clone https://github.com/probcomp/b3d.git
	pushd b3d
	git checkout eightysteele/gen-389-repro-environment
	popd
fi

echo "b3d bootstrapped."
echo "change into the b3d directory and install..."
echo "cd b3d && ./install.sh"
