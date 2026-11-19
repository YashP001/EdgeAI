#!/usr/bin/env bash
# Run from the Codespace terminal: bash tools/start_codespace.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f model/model.pt || ! -f model/config.json ]]; then
  echo 'Missing trained model. Run python train.py --team 4 --sweep first.' >&2
  exit 1
fi
python -c 'import torch, fastapi, uvicorn, numpy' 2>/dev/null || python -m pip install -r requirements.txt
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
