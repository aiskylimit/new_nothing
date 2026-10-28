"""Unpacks one folder, for folders added after a server unpacked an older processed_data.zip:
    python tools/unpack_data.py mt_v2   ->  processed_data/general_domain/mt_v2/
Takes it from <folder>.zip (its own download.txt entry, e.g. mt_v2.zip, so an old processed_data.zip already
on the server can't stand in for it), else from processed_data.zip. Fails when neither has the folder."""
import os
import sys
import zipfile

name = sys.argv[1]
prefix = f"processed_data/general_domain/{name}/"
for path in (f"{name}.zip", "processed_data.zip"):
    if not os.path.exists(path):
        continue
    z = zipfile.ZipFile(path)
    members = [n for n in z.namelist() if n.startswith(prefix)]
    if members:
        z.extractall(".", members)
        print(f"unpacked {len(members)} files from {path} into {prefix}")
        sys.exit(0)
sys.exit(f"no {prefix} in {name}.zip or processed_data.zip: fetch {name}.zip with download.txt")
