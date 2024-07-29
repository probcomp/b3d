![b3d](https://github.com/probcomp/b3d/assets/66085644/50bc2fc3-c9cd-4139-bed6-9c6d53933622)

<p align="center">
  <img src="https://github.com/probcomp/b3d/assets/66085644/53d4f644-530e-41b9-87f9-814064f12230" alt="animated" height="300" />
</p>

# B3D: Bayesian 3D Inverse Graphics

## What is B3D?

B3D is a scene perception system[^1] that efficiently learns 3D object models in real-time, with robust uncertainty estimation, using GPU-accelerated sequential Monte Carlo inference.

## Requirements

B3D requires a `linux-64` machine and a CUDA-enabled NVIDIA GPU[^2]. It also requires access to GenJAX, which you can get by sending this to any channel on the MIT Probcomp Slack: `\invite-genjax <google-account-email>`

> [!TIP]
> B3D supports a `cpu` environment for `osx-arm64` machines. This environment provides tools for managing and connecting to B3D VMs on Google Cloud Platform.

## Setup

The B3D environment includes developer tools, Python packages, and system-level dependencies that get pinned together in a single lockfile, with exact package versions, for each supported platform (`linux-64` and `osx-arm64`) across all virtual environments (`gpu` and `cpu`). This makes it possible to reproduce B3D on any machine.

### Bootstrap

Before installing B3D, you bootstrap the environment. This will install `pixi`[^3] and `git` (if needed), update your shell configuration, and clone the `b3d` repo:

```sh
curl -fsSL https://raw.githubusercontent.com/probcomp/b3d/aaron/pixi/install/bootstrap.sh | bash
source ~/.bashrc
```

> [!TIP]
> The process can be customized using the following variables:
>
> | Variable            | Description                         | Default  |
> | :---                | :---                                |:---      |
> | B3D_CLONE           | clone the b3d repo                  | yes      |
> | B3D_HOME            | where to clone the b3d repo         | $PWD     |
> | B3D_BRANCH          | branch to checkout                  | main     |
> | B3D_CLONE_METHOD    | git clone method (https or ssh)     | https    |

### Install

After bootstrapping, you install B3D. This will automatically detect your platform, install the correct environments and dependencies, and prompt you for Google Cloud authentication:

```sh
cd b3d
pixi run b3d-install
```

> [!TIP]
> If `pixi run b3d-install` fails because it can't find `pixi`, you may have forgotten to source your shell after bootstrapping. To fix it, run `source ~/.bashrc` and try installing again.

## Developing

You can install B3D on a GPU-enabled VM and connect to it from `vscode` on your local machine using `gcp` tasks.

### Setup

To setup `vscode`, you need to install a couple of extensions and make sure the `code` command is available on your local machine.

1. Install the [Remote Development Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) (for connecting to VMs over SHH)
2. Install the latest [Microsoft Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) (this provides `pixi` environment support by selecting the `pre-release` version of the extension)
3. On `osx-arm64`, run `Install code command in PATH` from the Command Palette so that `code` is available on your local machine


### GCP

`gcp` tasks manage VMs, reserve static IP addresses, and update ssh configuration files (e.g., host entries, remote forwarding for `rerun`, etc.). To create a new VM and connect to it through `vscode`:

```shell
export GCP_VM=neyman-b3d-gpu
pixi run gcp-code
```

For more `gcp` tasks and options, see `pixi task list` and `pixi run gcp-help`.

### Tests

Before running tests, this task will prompt you to synchronize test data, delete the `pytorch` extension cache, and launch `rerun`.

```shell
pixi run dev-tests
```

## Footnotes

[^1]: https://arxiv.org/pdf/2312.08715
[^2]: The NVIDIA GPU must have a compute capability of `5.2` or newer, and the NVIDIA driver must be version `525.60.13` or newer.
[^3]: https://github.com/prefix-dev/pixi/?tab=readme-ov-file#overview
