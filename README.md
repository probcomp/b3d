![b3d](https://github.com/probcomp/b3d/assets/66085644/50bc2fc3-c9cd-4139-bed6-9c6d53933622)

<p align="center">
  <img src="https://github.com/probcomp/b3d/assets/66085644/53d4f644-530e-41b9-87f9-814064f12230" alt="animated" width="150%" />
</p>

## Requirements
- NVIDIA CUDA 12.3+
- Python 3.10+

## Installing b3d
```
./install.sh
```


### Installing CUDA 12.3+
Run `nvidia-smi`
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
```
and verify that `CUDA Version: xx.x` is >= 12.3.

If not, ensure your NVIDIA GPU supports CUDA 12.3+ and install a NVIDIA driver version compatible with 12.3+. For GCP instances, use:
```
sudo sh -c "echo 'export DRIVER_VERSION=550.54.15' > /opt/deeplearning/driver-version.sh"
/opt/deeplearning/install-driver.sh
```
If the above commands fail, then first uninstall the existing driver by running the below, then try again.
```
/opt/deeplearning/uninstall-driver.sh
sudo reboot
```

### Environment Variables
Add the following to your bash_rc:
```
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
```


### Visualizer
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
