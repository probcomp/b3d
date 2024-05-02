filename="$1_google_16k.tgz"
"Downloading additional ycb models: $1"
wget "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/$filename"
tar -vxzf $filename -C assets/shared_data_bucket/ycb_video_models/models
rm $filename
mv "assets/shared_data_bucket/ycb_video_models/models/$1/google_16k"/* "assets/shared_data_bucket/ycb_video_models/models/$1"
rm -r "assets/shared_data_bucket/ycb_video_models/models/$1/google_16k/"

