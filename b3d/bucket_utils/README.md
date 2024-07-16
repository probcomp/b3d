## Interacting with the GCP Bucket

### Get access
Contact one of the existing B3D GCP Bucket admins to get access to [gs://b3d_bucket](https://console.cloud.google.com/storage/browser/b3d_bucket).


### Download data
The `b3d_pull` command is used for downloading data from the GCP bucket into your local.


##### Download all bucket data (disable overwrite)
By default, the `b3d_pull` command pulls _all existing data_ in the GCP bucket into `b3d/assets/shared_bucket_data`. If the local directory does not already exist, it will automatically be initialized.
By default, files with names that already exist in your local `b3d/assets/shared_bucket_data` will not be overwritten with the version on the cloud, i.e. the download will be skipped for those files.

```
b3d_pull   # pulls all bucket contents into local, 
            with no overwrite on already-existing filenames
```

##### Download all bucket data (enable overwrite)
To pull _all existing data_ in the GCP bucket _with overwrite_ into your local `b3d/assets/shared_bucket_data`, use the `-ow` flag. 
Note that other than potential overwriting behavior enabled by `-ow`, `b3d_pull` never deletes files on your local directory; be sure to delete deprecated local files before `b3d_push`'ing your local contents onto the bucket.

```
b3d_pull -ow  # pulls all bucket contents into local, 
                with overwrite on already-existing filenames
```

##### Download select bucket data items
Finally, to pull select items (files, directories) in the GCP bucket into your local `b3d/assets/shared_bucket_data`, use the `-fn` flag, followed by the path of the item relative to the root of the bucket.
All necessary intermediate directories will automatically be initialized if not already existing in your local `b3d/assets/shared_bucket_data`. 

For example, if you would like to pull `input_data/banana_datas/banana1.npz`, 

```
b3d_pull -fn input_data/banana_datas/banana1.npz  # populates dcolmap/assets/shared_bucket_data/input_data/banana_datas/banana1.npz
```

For another example, if you would like to pull all items in `input_data/mug_datas`, 

```
b3d_pull -fn input_data/mug_datas  # recursively populates dcolmap/assets/shared_bucket_data/input_data/mug_datas
```

You can combine these flags. For example, if you would like to pull all items in `input_data/mug_datas` with overwrite enabled, 

```
b3d_pull -ow -fn input_data/mug_datas  # recursively populates dcolmap/assets/shared_bucket_data/input_data/mug_datas, 
                                            with overwrite on already-existing filenames
```

### Upload data
The `b3d_push` command is used for downloading data from your local into the GCP bucket. The `-ow` and `-fn` flags behave analogously as in the `b3d_pull` case.

##### Upload all local data (disable overwrite)
```
b3d_push   # pushes all local content onto bucket, 
            with no overwrite on already-existing filenames
```

##### Upload all local data (enable overwrite)
```
b3d_push -ow  # pushes all local content onto bucket, with overwrite. A DANGEROUS COMMAND (triggers y/n confirmation)
```

##### Upload select local data items
```
b3d_push -ow -fn input_data/mug_datas  # recursively populates gs://b3d_bucket/input_data/mug_datas, 
                                            with overwrite on already-existing filenames
```
