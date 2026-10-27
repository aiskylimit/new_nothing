#! /bin/bash
# Packs every saved checkpoint under results/ into zip files of at most 24 MB (24,000,000 bytes) each, in
# checkpoint/ with the same layout: results/<run>/<settings>/<step>/ -> checkpoint/<run>/<settings>/<step>/
#   part_000.zip, part_001.zip, ...   the checkpoint folder as one tar, cut into 23,000,000-byte pieces, each
#                                     piece zipped on its own (adapter weights barely compress, so a zip stays
#                                     close to 23 MB, never above 24 MB)
#   ckpt.tar.sha256                   sha256 of the whole tar
# A checkpoint = a <step>/ folder with adapter_config.json (LoRA teacher) or config.json (full student).
# Already packed checkpoints (ckpt.tar.sha256 present) are skipped, so rerun it after new runs finish. Don't
# run it while a job is still saving: a half-written checkpoint would be packed as it is.
# At the end every file in checkpoint/ is checked: any file over 24,000,000 bytes makes the script fail.
#
# Restore one checkpoint (gives <step>/ in the current folder; do it on a copy outside checkpoint/, or the
# size check of the next run fails on the restored files):
#   cd <copy of checkpoint/<run>/<settings>/<step>>
#   python3 -c "import glob, zipfile; [zipfile.ZipFile(z).extractall('.') for z in sorted(glob.glob('part_*.zip'))]"
#   cat part_??? > ckpt.tar && sha256sum -c ckpt.tar.sha256 && tar -xf ckpt.tar && rm part_??? ckpt.tar
cd "$(dirname "$0")"

MAX_BYTES=24000000
PART_BYTES=23000000
OUT=checkpoint

if [ ! -d results ]; then
    echo "no results/ folder here"
    exit 1
fi

# vllm_tmp/ holds the model copies vLLM evaluates, not checkpoints
for cfg in $(find results -path '*/vllm_tmp' -prune -o -type f \( -name adapter_config.json -o -name config.json \) -print | sort); do
    ckpt=$(dirname "${cfg}")
    [[ $(basename "${ckpt}") =~ ^[0-9]+$ ]] || continue
    dest=${OUT}/${ckpt#results/}
    if [ -f "${dest}/ckpt.tar.sha256" ]; then
        echo "skip (packed): ${ckpt}"
        continue
    fi
    rm -rf "${dest}"
    mkdir -p "${dest}"
    tar -cf - -C "$(dirname "${ckpt}")" "$(basename "${ckpt}")" | split -b ${PART_BYTES} -d -a 3 - "${dest}/part_" || exit 1
    # the image has no zip binary, so Python's zipfile zips each piece (then the raw piece is removed)
    python3 - "${dest}" <<'EOF' || exit 1
import glob, os, sys, zipfile
for part in sorted(glob.glob(os.path.join(sys.argv[1], "part_[0-9][0-9][0-9]"))):
    with zipfile.ZipFile(part + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(part, os.path.basename(part))
    os.remove(part)
EOF
    # hash of the tar the pieces rebuild: unzip them again and hash them in order
    python3 - "${dest}" <<'EOF' || exit 1
import glob, hashlib, os, sys, zipfile
h = hashlib.sha256()
for z in sorted(glob.glob(os.path.join(sys.argv[1], "part_*.zip"))):
    with zipfile.ZipFile(z) as f, f.open(f.namelist()[0]) as part:
        for block in iter(lambda: part.read(1 << 20), b""):
            h.update(block)
with open(os.path.join(sys.argv[1], "ckpt.tar.sha256"), "w", newline="\n") as f:
    f.write(f"{h.hexdigest()}  ckpt.tar\n")
EOF
    n=$(ls "${dest}"/part_*.zip | wc -l)
    echo "packed ${ckpt}: $(du -sh "${ckpt}" | cut -f1) -> ${n} zips, $(du -sh "${dest}" | cut -f1)"
done

# size check: nothing in checkpoint/ may be larger than 24,000,000 bytes
big=$(find ${OUT} -type f -size +${MAX_BYTES}c 2>/dev/null)
if [ -n "${big}" ]; then
    echo "ERROR: files over ${MAX_BYTES} bytes:"
    for f in ${big}; do echo "  $(stat -c %s "${f}") ${f}"; done
    exit 1
fi
echo "size check ok: $(find ${OUT} -type f -name 'part_*.zip' | wc -l) zips in ${OUT}/, largest $(find ${OUT} -type f -printf '%s\n' | sort -n | tail -1) bytes (limit ${MAX_BYTES})"
