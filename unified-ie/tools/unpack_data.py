"""Unpacks one folder from processed_data.zip, for folders added after a server unpacked an older zip:
    python tools/unpack_data.py mt_v2   ->  processed_data/general_domain/mt_v2/
Fails when the zip has no such folder (an old zip: fetch it again with download.txt)."""
import sys
import zipfile

prefix = f"processed_data/general_domain/{sys.argv[1]}/"
z = zipfile.ZipFile("processed_data.zip")
members = [n for n in z.namelist() if n.startswith(prefix)]
if not members:
    sys.exit(f"processed_data.zip has no {prefix}: delete it and fetch it again with download.txt")
z.extractall(".", members)
print(f"unpacked {len(members)} files into {prefix}")
