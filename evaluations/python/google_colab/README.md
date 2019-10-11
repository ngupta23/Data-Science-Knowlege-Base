#Uploading to and saving data in Google Drive using Colab
https://course.fast.ai/start_colab.html

```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'fastai-v3/'
```

```
path = Path(base_dir + 'data/images')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
```


#Uploading to Google Colab from local computer
https://towardsdatascience.com/3-ways-to-load-csv-files-into-colab-7c14fcbdcb92
```
from google.colab import files
uploaded = files.upload()
# This will prompy
```

#Uploading data to Colab using GitHub
https://medium.com/@yuraist/how-to-upload-your-own-dataset-into-google-colab-e228727c87e9
* Will clone data into colab VM instance
* Any modifications to this data will be lost once the colab notebook is closed.