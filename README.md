![b3d](https://github.com/probcomp/b3d/assets/66085644/50bc2fc3-c9cd-4139-bed6-9c6d53933622)

<p align="center">
  <img src="https://github.com/probcomp/b3d/assets/66085644/53d4f644-530e-41b9-87f9-814064f12230" alt="animated" width="150%" />
</p>

# b3d

This repository contains code for Bayesian 3D inverse graphics.

The `b3d/bayes3d/` subdirectory contains code for the `bayes3d` project, and the `b3d/chisight/` subdirectory contains code for post-bayes3D ChiSight systems (currently, SAMA4D).

## GCloud Machine

Run the following command to launch your instance:

```bash
export ZONE="us-west1-a"

# Make sure to replace the value with a unique name!
export INSTANCE_NAME="your-instance-name-here"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family="common-cu123-ubuntu-2204-py310" \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=400GB \
  --machine-type g2-standard-32 \
  --accelerator="type=nvidia-l4,count=1" \
  --metadata="install-nvidia-driver=True"
```

to ssh into your new machine, run:

```bash
gcloud compute config-ssh
ssh $INSTANCE_NAME.$ZONE.$PROJECT_ID
```

## Environment Variables

Add the following to your bash_rc:

```sh
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
```

# Installing b3d

`b3d`'s GenJAX dependency is hosted in Google Artifact Registry. To configure your machine to access the package:

- Check if you can access the [GenJAX Users
  Group](https://groups.google.com/u/1/a/chi-fro.org/g/genjax-users), and if not, run `\invite-genjax <google-account-email>` in any channel in the the probcomp Slack
- [Install the Google Cloud command line tools](https://cloud.google.com/sdk/docs/install).
- Run `gcloud auth application-default login`.  (This command needs to be rerun ever time your machine reboots.)

Next, on the machine, close the `b3d` repo:

```sh
git clone https://github.com/probcomp/b3d.git
cd b3d
```

Create and activate `b3d` conda environment:

```sh
conda create -n b3d python=3.12
```

When that completes, run the install script:

```sh
bash -i install.sh
```

Then activate the conda environment:

```sh
conda activate b3d
```

## Visualizer

Tunnel port `8812` for Rerun visualization by adding the `RemoteForward`line to your ssh config:

```sh
Host xx.xx.xxx.xxx
    HostName xx.xx.xxx.xxx
    IdentityFile ~/.ssh/id_rsa
    User thomasbayes
    RemoteForward 8812 127.0.0.1:8812
```

Install rerun on local machine `pip install rerun-sdk` and open viewer:

```sh
rerun --port 8812
```
