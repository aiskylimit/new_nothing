from pathlib import Path
import site
import sys


RELATIVE_FILE = "transformers/models/qwen2_vl/image_processing_qwen2_vl.py"
START_LINE = 140
END_LINE = 143


def find_transformers_file():
    site_package_dirs = site.getsitepackages()

    user_site = site.getusersitepackages()
    if user_site:
        site_package_dirs.append(user_site)

    for site_package_dir in site_package_dirs:
        file_path = Path(site_package_dir) / RELATIVE_FILE
        if file_path.exists():
            return file_path

    return None


file_path = find_transformers_file()
if file_path is None:
    print(
        "Could not find transformers qwen2_vl image processor. "
        "Make sure the venv is activated and requirements are installed.",
        file=sys.stderr,
    )
    sys.exit(1)

with file_path.open("r") as f:
    lines = f.readlines()

with file_path.open("w") as f:
    for line_number, line in enumerate(lines, start=1):
        if START_LINE <= line_number <= END_LINE:
            if not line.lstrip().startswith("#"):
                f.write("# " + line)
            else:
                f.write(line)
        else:
            f.write(line)

print(f"Done. Lines {START_LINE}-{END_LINE} have been commented in {file_path}.")
