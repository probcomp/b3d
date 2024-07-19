# Developer's Guide

This guide describes how to complete various tasks you'll encounter when working
on the b3d codebase.

## Commit Hooks

We use [pre-commit](https://pre-commit.com/) to manage a series of git pre-commit hooks for the
project; for example, each time you commit code, the hooks will make sure that your python is
formatted properly. If your code isn't, the hook will format it, so when you try to commit the
second time you'll get past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks, install `pre-commit` if
you don't yet have it. I prefer using [pipx](https://github.com/pipxproject/pipx) so that
`pre-commit` stays globally available.

```bash
pipx install pre-commit
```

Then install the hooks with this command:

```bash
pre-commit install
```

Now they'll run on every commit. If you want to run them manually, run the following command:

```bash
pre-commit run --all-files
```

## Installing CUDA 12.3+

These instructions are for older gcloud machines than the one specified in README.md. If you created an instance using those instructions, you should _not_ have to do this.

To diagnose errors from an older-than-12.3 CUDA version, run `nvidia-smi`:

```sh
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
```

and verify that `CUDA Version: xx.x` is >= 12.3.

If not, ensure your NVIDIA GPU supports CUDA 12.3+ and install a NVIDIA driver version compatible with 12.3+. For GCP instances, use:

```sh
sudo sh -c "echo 'export DRIVER_VERSION=550.54.15' > /opt/deeplearning/driver-version.sh"
/opt/deeplearning/install-driver.sh
```

If the above commands fail, then first uninstall the existing driver by running the below, then try again.

```sh
/opt/deeplearning/uninstall-driver.sh
sudo reboot
```
