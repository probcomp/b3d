export SRC=https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main
wget $SRC/ycbv/ycbv_base.zip -P assets/bop         # Base archive with dataset info, camera parameters, etc.
wget $SRC/ycbv/ycbv_models.zip  -P assets/bop      # 3D object models.
wget $SRC/ycbv/ycbv_test_all.zip -P assets/bop    # All test images ("_bop19" for a subset used in the BOP Challenge 2019/2020).

unzip assets/bop/ycbv_base.zip -d assets/bop
unzip assets/bop/ycbv_models.zip -d assets/bop/ycbv
unzip assets/bop/ycbv_test_all.zip -d assets/bop/ycbv
