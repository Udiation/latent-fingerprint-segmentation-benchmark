# Public release procedure

This procedure moves only aggregate, receipt-authenticated Stage-5 outputs into
the public repository. It never imports per-image records, biometric media,
checkpoints, probability maps, or trained weights.

## 1. Dry conversion

Obtain the SHA-256 of `package_receipt.json` through an independent channel.
Do not compute the expected value inside the import command.

```bash
python scripts/import_stage5_public.py \
  --package "$STAGE5_PACKAGE" \
  --expected-receipt-sha256 "$STAGE5_RECEIPT_SHA256" \
  --require-release-metadata --dry-run \
  --report /path/outside/the/repository/import-dry-run.json
```

The converter verifies the complete `public/` file universe and every file
digest against the canonical receipt. It checks the statistics CSV and JSON for
semantic identity, converts tables without hand-entered values, validates the
receipt-bound `release_universe.json`, chosen configs, and 136/24/40 split, and
stages the website before reporting any success. It rejects incomplete method
or examiner-group universes and preserves
the canonical across-seed standard deviations in every generated result view.

## 2. Controlled apply

The apply starts only from a clean Git worktree. It creates a deliberate dirty
tree containing the generated public outputs and a safe import receipt.

```bash
python scripts/import_stage5_public.py \
  --package "$STAGE5_PACKAGE" \
  --expected-receipt-sha256 "$STAGE5_RECEIPT_SHA256" \
  --require-release-metadata --apply \
  --report /path/outside/the/repository/import-apply.json
python scripts/check_site_numbers.py
python scripts/check_release_requirements.py
python scripts/scan_release.py --phase working \
  --stage5-package "$STAGE5_PACKAGE" \
  --expected-stage5-receipt-sha256 "$STAGE5_RECEIPT_SHA256" \
  --out /path/outside/the/repository/working-scan.json
```

The working phase deliberately permits the controlled dirty tree. Review and
commit exactly those outputs; do not edit a generated number or figure.

## 3. Final clean-tree gates

```bash
python scripts/scan_release.py --phase final \
  --stage5-package "$STAGE5_PACKAGE" \
  --expected-stage5-receipt-sha256 "$STAGE5_RECEIPT_SHA256" \
  --out /path/outside/the/repository/final-scan.json
lighthouse https://udiation.github.io/latent-fingerprint-segmentation-benchmark/ \
  --only-categories=accessibility --output=json \
  --output-path=/path/outside/the/repository/lighthouse.json
python scripts/check_lighthouse.py \
  --input /path/outside/the/repository/lighthouse.json \
  --out /path/outside/the/repository/lighthouse-gate.json \
  --expected-url https://udiation.github.io/latent-fingerprint-segmentation-benchmark/ \
  --maximum-age-hours 24 \
  --minimum 0.90
```

The final scan fails unless the worktree and HEAD remain unchanged throughout
the scan and the report path is outside the repository. Aggregate PNG/PDF plots
are accepted only when declared by `figures/figure_manifest.json`, with exact
path, SHA-256, MIME, semantic plot kind, and canonical source-receipt identity.
The final scan re-derives every imported output from that source package, so a
joint edit of a figure and its local manifest cannot create a false pass. The
Lighthouse gate also checks its requested/final URL, timestamp, and hashes of
the current site index and build manifest, rejecting a stale or unrelated run.
