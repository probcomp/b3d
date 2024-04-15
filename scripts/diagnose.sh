nvidia-smi
python -c "import torch; print('Torch version :', torch.__version__); print('Torch CUDA available: ', torch.cuda.is_available())"
python -c "import jax; print('JAX Version: ',jax.__version__); print('JAX Devices :', jax.devices())"