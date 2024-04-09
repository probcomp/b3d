![b3d](https://github.com/probcomp/b3d/assets/66085644/50bc2fc3-c9cd-4139-bed6-9c6d53933622)

<p align="center">
  <img src="https://github.com/probcomp/b3d/assets/66085644/53d4f644-530e-41b9-87f9-814064f12230" alt="animated" width="150%" />
</p>



## Installation
### NVIDIA GPU with CUDA 12.3+
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

### Python Environment
```
conda create -n b3d python=3.10
conda activate b3d
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

### Install b3d

Run
```
bash install.sh
```
Verify install succeeded by running `python demo.py` which should print:
```
FPS: 175.81572963059995
```
and display a `demo.py` visualization log in Rerun viewer that shows data corresponding to the gif at the top of this README!

