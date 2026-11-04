# """Copies the best dev checkpoint of teacher SFT runs to the private HF dataset repo datht/unified-ie, so a
# server reset does not cost the teachers again.

#     python tools/upload_teachers.py results/qwen3/general_domain/sft_4B_ee_v2_ace ...

# For each run: the LoRA dir tools/best_lora.py picks, plus log.txt and args.json of its <settings> dir, to
# teachers/<run>/<settings>/ in the repo. Token: HF_TOKEN, else the cached `huggingface-cli login`; with
# neither it prints what it skipped and exits 0 (the queue goes on).
# To get them back: huggingface-cli download datht/unified-ie --repo-type dataset --include "teachers/*"
# --local-dir hf and move hf/teachers/<run> to results/qwen3/general_domain/<run>."""
# import os
# import subprocess
# import sys

# from huggingface_hub import HfApi
# from huggingface_hub.utils import get_token

# REPO = "datht/unified-ie"

# token = os.environ.get("HF_TOKEN") or get_token()
# if not token:
#     print(f"no HF_TOKEN and no cached login: not uploading {' '.join(sys.argv[1:])}")
#     sys.exit(0)
# api = HfApi(token=token)

# for run in sys.argv[1:]:
#     run = run.rstrip("/")
#     best = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "best_lora.py"), run],
#                           capture_output=True, text=True)
#     if best.returncode != 0:
#         print(f"{run}: no best checkpoint, not uploaded\n{best.stderr}")
#         continue
#     lora = best.stdout.strip()
#     settings = os.path.dirname(lora)
#     dest = f"teachers/{os.path.basename(run)}/{os.path.basename(settings)}"
#     api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=lora,
#                       path_in_repo=f"{dest}/{os.path.basename(lora)}",
#                       commit_message=f"teacher {os.path.basename(run)}: best dev checkpoint {os.path.basename(lora)}")
#     for f in ("log.txt", "args.json"):
#         if os.path.exists(os.path.join(settings, f)):
#             api.upload_file(repo_id=REPO, repo_type="dataset", path_or_fileobj=os.path.join(settings, f),
#                             path_in_repo=f"{dest}/{f}", commit_message=f"teacher {os.path.basename(run)}: {f}")
#     print(f"{run}: uploaded {lora} to {REPO}/{dest}")
