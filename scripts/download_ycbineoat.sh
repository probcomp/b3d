export SRC=https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCBInEOAT/
mkdir -p assets/ycbineoat

files=("bleach0.tar.gz" "bleach_hard_00_03_chaitanya.tar.gz" "cracker_box_reorient.tar.gz" "cracker_box_yalehand0.tar.gz" "mustard0.tar.gz" "mustard_easy_00_02.tar.gz" "sugar_box1.tar.gz" "sugar_box_yalehand0.tar.gz" "tomato_soup_can_yalehand0.tar.gz")
for i in ${files[@]}; do
  wget $SRC$i -P assets/ycbineoat
done

for i in ${files[@]}; do
  tar -xzvf assets/ycbineoat/$i -C assets/ycbineoat
done
