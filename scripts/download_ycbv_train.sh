export SRC=https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main
wget $SRC/ycbv/ycbv_train_real.zip -P assets/bop
wget $SRC/ycbv/ycbv_train_real.z01 -P assets/bop
zip -s0 assets/bop/ycbv_train_real.zip --out assets/bop/ycbv_train_real_all.zip
unzip assets/bop/ycbv_train_real_all.zip -d assets/bop/ycbv
