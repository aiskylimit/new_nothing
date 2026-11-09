"""Run ONLY on an internet-connected provisioning host, then copy assets to B200."""
import argparse
import json
import os
import urllib.request
from pathlib import Path

from rebuttal.common import DATA_REV, GEMINI_REV, MODELS, sha, write

BENCHMARKS = {
    'AIME_2024': ('Maxwell-Jia/AIME_2024', '8d88b2876a82a080e2f172cc9b25d0d9d2cb4792'),
    'aime_2025': ('yentinglin/aime_2025', '6f71d77b0b89b9dabe07ab466c51df33f514df7f'),
    'aimo-validation-amc': ('AI-MO/aimo-validation-amc', '69d78a4a2c840e82d69af6bc742bda09005f6316'),
    'MATH-500': ('HuggingFaceH4/MATH-500', '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'),
}
URLS = {
    'ARC-Challenge/test-00000-of-00001.parquet': (
        'https://huggingface.co/datasets/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/test-00000-of-00001.parquet',
        '62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9'),
    'LogiQA2.0/test.txt': ('https://raw.githubusercontent.com/csitfun/LogiQA2.0/31381f5634d201c4ed6f981b87de3c7d9f496002/logiqa/DATA/LOGIQA/test.txt',
                          '71940b37ae0184b677c253a148d57ad4e75d6113447b1563c2ca82483e4e4f8d'),
}


def attest(folder, repo, revision):
    files = sorted(p for p in folder.rglob('*') if p.is_file() and '.cache' not in p.parts and p.name != 'snapshot.json')
    write(folder / 'snapshot.json', {'repo': repo, 'revision': revision,
        'sha256': {str(p.relative_to(folder)): sha(p) for p in files}})


def verify_snapshot(folder, repo, revision):
    metadata = json.loads((folder / 'snapshot.json').read_text())
    if metadata['revision'] != revision or metadata['repo'] != repo:
        raise ValueError(f'Wrong snapshot identity: {folder}')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and '.cache' not in p.parts and p.name != 'snapshot.json'}
    if actual != set(metadata['sha256']):
        raise ValueError(f'Snapshot file set changed: {folder}')
    for relative, expected in metadata['sha256'].items():
        if sha(folder / relative) != expected:
            raise ValueError(f'Snapshot content changed: {folder / relative}')
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--attest-managed', action='store_true', help='Offline: verify Hub local-dir commit metadata, then attest existing files')
    args = p.parse_args()
    specs = [(repo, revision, 'model', 'models/'+repo.split('/')[-1]) for repo, revision in MODELS.values()]
    specs += [('simplescaling/s1K-1.1', DATA_REV, 'dataset', 'datasets/s1K-1.1'),
              ('simplescaling/s1K', GEMINI_REV, 'dataset', 'datasets/s1K')]
    specs += [(repo, revision, 'dataset', 'benchmarks/'+name) for name, (repo, revision) in BENCHMARKS.items()]
    for repo, revision, kind, relative in specs:
        folder = args.assets / relative
        if args.attest_managed:
            metadata = list((folder / '.cache/huggingface/download').rglob('*.metadata'))
            if not metadata or any(path.read_text().splitlines()[0] != revision for path in metadata):
                raise ValueError(f'{folder}: missing/mismatched Hub commit metadata; provision the pinned snapshot online first')
            for file in folder.rglob('*'):
                if file.is_file() and '.cache' not in file.parts and file.name != 'snapshot.json':
                    record = folder / '.cache/huggingface/download' / (str(file.relative_to(folder))+'.metadata')
                    if not record.exists() or record.read_text().splitlines()[0] != revision:
                        raise ValueError(f'Unattested managed-download file: {file}')
        else:
            from huggingface_hub import snapshot_download
            if folder.exists() and any(folder.iterdir()):
                verify_snapshot(folder, repo, revision)
                continue
            # Stage by immutable commit so interrupted downloads can resume
            # without mixing files left by an earlier model/dataset revision.
            staging = args.assets / '.staging' / (repo.replace('/', '--') + '_' + revision)
            snapshot_download(repo, repo_type=kind, revision=revision, local_dir=staging)
            attest(staging, repo, revision)
            folder.parent.mkdir(parents=True, exist_ok=True)
            if folder.exists():
                folder.rmdir()  # already checked empty above
            os.replace(staging, folder)
            continue
        attest(folder, repo, revision)
    for relative, (url, expected) in URLS.items():
        path = args.assets / 'benchmarks' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not args.attest_managed and not path.exists():
            urllib.request.urlretrieve(url, path)
        if sha(path) != expected:
            raise ValueError(f'OOD file checksum mismatch: {path}')


if __name__ == '__main__':
    main()
