import json
import os
from concurrent.futures import ThreadPoolExecutor
import urllib.request as ureq

with open('dataset.json', 'r') as fp:
    data = json.load(fp)

os.makedirs("images", exist_ok=True)

def download_image(k):
    url = data[k]['imageURL']
    ext = os.path.splitext(url)[1]
    output = f'images/{k}{ext}'

    if not os.path.exists(output):
        try:
            ureq.urlretrieve(url, output)
            print(f"Downloaded {k}")
        except Exception as e:
            print(f"Failed {k}: {e}")

keys = list(data.keys())

with ThreadPoolExecutor(max_workers=32) as ex:
    ex.map(download_image, keys)