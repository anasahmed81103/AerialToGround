"""
Upload Stage-1 + Stage-2 checkpoints to a Hugging Face model repo.

  pip install huggingface_hub
  huggingface-cli login
  python upload_to_hf.py --repo_id YOUR_USER/aerial2ground-cvusa
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi, create_repo

DEFAULT_FILES = [
    ('experiments/phase2/runs/polar_full/best.pt', 'polar_full/best.pt'),
    (
        'experiments/phase2/runs/spade_aerial_v2/ckpts/G_step0032000.pt',
        'spade_aerial_v2/G_step0032000.pt',
    ),
    ('experiments/phase2/runs/polar_full/config.json', 'polar_full/config.json'),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--repo_id', required=True, help='HF repo id, e.g. user/aerial2ground')
    p.add_argument('--private', action='store_true')
    args = p.parse_args()

    api = HfApi()
    create_repo(args.repo_id, exist_ok=True, private=args.private)

    for local, remote in DEFAULT_FILES:
        if not os.path.isfile(local):
            raise SystemExit(f'Missing checkpoint: {local}')
        print(f'[*] upload {local} -> {remote}', flush=True)
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type='model',
        )
    print(f'[*] Done: https://huggingface.co/{args.repo_id}', flush=True)


if __name__ == '__main__':
    main()
