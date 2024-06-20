## Interacting with the GCP Bucket

### Get access
Contact one of the existing HGPS GCP Bucket admins to get access to [gs://hgps_data_bucket](https://console.cloud.google.com/storage/browser/hgps_data_bucket).


### Download data
The `chi_pull` command is used for downloading data from the GCP bucket into your local.


##### Download all bucket data (disable overwrite)
By default, the `chi_pull` command pulls _all existing data_ in the GCP bucket into `dcolmap/assets/shared_bucket_data`. If the local directory does not already exist, it will automatically be initialized.
By default, files with names that already exist in your local `dcolmap/assets/shared_bucket_data` will not be overwritten with the version on the cloud, i.e. the download will be skipped for those files.

```
chi_pull   # pulls all bucket contents into local, 
            with no overwrite on already-existing filenames
```

##### Download all bucket data (enable overwrite)
To pull _all existing data_ in the GCP bucket _with overwrite_ into your local `dcolmap/assets/shared_bucket_data`, use the `-ow` flag. 
Note that other than potential overwriting behavior enabled by `-ow`, `chi_pull` never deletes files on your local directory; be sure to delete deprecated local files before `chi_push`'ing your local contents onto the bucket.

```
chi_pull -ow  # pulls all bucket contents into local, 
                with overwrite on already-existing filenames
```

##### Download select bucket data items
Finally, to pull select items (files, directories) in the GCP bucket into your local `dcolmap/assets/shared_bucket_data`, use the `-fn` flag, followed by the path of the item relative to the root of the bucket.
All necessary intermediate directories will automatically be initialized if not already existing in your local `dcomap/assets/shared_bucket_data`. 

For example, if you would like to pull `input_data/banana_datas/banana1.npz`, 

```
chi_pull -fn input_data/banana_datas/banana1.npz  # populates dcolmap/assets/shared_bucket_data/input_data/banana_datas/banana1.npz
```

For another example, if you would like to pull all items in `input_data/mug_datas`, 

```
chi_pull -fn input_data/mug_datas  # recursively populates dcolmap/assets/shared_bucket_data/input_data/mug_datas
```

You can combine these flags. For example, if you would like to pull all items in `input_data/mug_datas` with overwrite enabled, 

```
chi_pull -ow -fn input_data/mug_datas  # recursively populates dcolmap/assets/shared_bucket_data/input_data/mug_datas, 
                                            with overwrite on already-existing filenames
```

### Upload data
The `chi_push` command is used for downloading data from your local into the GCP bucket. The `-ow` and `-fn` flags behave analogously as in the `chi_pull` case.

##### Upload all local data (disable overwrite)
```
chi_push   # pushes all local content onto bucket, 
            with no overwrite on already-existing filenames
```

##### Upload all local data (enable overwrite)
```
chi_push -ow  # pushes all local content onto bucket, with overwrite. A DANGEROUS COMMAND (triggers y/n confirmation)
```

##### Upload select local data items
```
chi_push -ow -fn input_data/mug_datas  # recursively populates gs://hgps_data_bucket/input_data/mug_datas, 
                                            with overwrite on already-existing filenames
```
