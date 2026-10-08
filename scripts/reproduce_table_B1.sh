#!/usr/bin/env bash
set -euo pipefail
exec python scripts/reproduce.py --table B1 "$@"

