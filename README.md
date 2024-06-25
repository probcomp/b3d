![b3d](https://github.com/probcomp/b3d/assets/66085644/50bc2fc3-c9cd-4139-bed6-9c6d53933622)

<p align="center">
  <img src="https://github.com/probcomp/b3d/assets/66085644/53d4f644-530e-41b9-87f9-814064f12230" alt="animated" width="150%" />
</p>

# b3d repository

This repository contains code for Bayesian 3D inverse graphics.

The `b3d/bayes3d/` subdirectory contains code for the `bayes3d` project, and the `b3d/chisight/` subdirectory contains code for post-bayes3D ChiSight systems (currently, SAMA4D).

## Installing CUDA 12.3+

Run `nvidia-smi`

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

## Installing b3d

`b3d`'s GenJAX dependency is hosted in Google Artifact Registry. To configure your machine to access the package:

- Check if you can access the [GenJAX Users Group](https://groups.google.com/u/1/a/chi-fro.org/g/genjax-users), and if not ask @sritchie (on Slack) to add you. Send him your Google-Cloud-associated email address.
- [Install the Google Cloud command line tools](https://cloud.google.com/sdk/docs/install).
- Run `gcloud auth application-default login`.  (This command needs to be rerun every time your machine reboots.)

When that completes, return to the `hgps` directory and create the `b3d` conda environment:

```sh
conda create -n b3d python=3.11
```

Then run the install script:

```sh
bash -i install.sh
```

### Environment Variables

Add the following to your bash_rc:

```sh
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
```

## Visualizer

Tunnel port `8812` for Rerun visualization by adding the `RemoteForward`line to your ssh config:

```
Host xx.xx.xxx.xxx
    HostName xx.xx.xxx.xxx
    IdentityFile ~/.ssh/id_rsa
    User thomasbayes
    RemoteForward 8812 127.0.0.1:8812
```

Install rerun on local machine `pip install rerun-sdk` and open viewer:
```
rerun --port 8812
```
