# Restricted demo deployment runbook

This runbook covers the private review application only. It does not authorize
copying a biometric artifact into Git, a public report, or a public site.

## Trust boundary

The process reads one receipt and only artifacts named by that receipt. The
receipt itself is authenticated by a SHA-256 supplied independently in the
environment. Every artifact has a relative path, byte count, type and SHA-256.
Absolute paths, parent traversal, symlinks, unknown artifact types and digest
mismatches fail before parsing. The test and B1 catalogues may contain test rows
only.

Build the receipt in restricted storage from a path-only spec:

```bash
python segbench/scripts/build_demo_receipt.py \
  --root "$DEMO_RESTRICTED_ROOT" \
  --spec "$DEMO_RESTRICTED_ROOT/demo/spec.json" \
  --out "$DEMO_RESTRICTED_ROOT/demo/receipt.json"
```

Record the printed digest through a separate operator-controlled channel. Do
not derive the pinned value from the receipt in the same launch command.

## Receipt schema

The path-only spec uses this shape. Artifact descriptors contain exactly
`path` and `kind`; the builder adds `bytes` and `sha256`.

```json
{
  "schema_version": 1,
  "kind": "segbench.restricted-demo-receipt.v1",
  "target_ppi": 500,
  "artifacts": {
    "test_manifest": {"path": "demo/test_manifest.csv", "kind": "csv"},
    "test_results": {"path": "demo/test_results.csv", "kind": "csv"},
    "test_predictions": {"path": "demo/test_predictions.csv", "kind": "csv"}
  },
  "methods": [
    {
      "key": "method_key",
      "display_name": "Display name",
      "reported_threshold": 0.4,
      "runner": {
        "type": "deep_checkpoint",
        "config_artifact": "method_config",
        "model_artifact": "method_checkpoint"
      }
    }
  ],
  "datasets": {
    "test": {
      "manifest": "test_manifest",
      "results": "test_results",
      "predictions": "test_predictions"
    },
    "b1": {
      "manifest": "b1_manifest",
      "results": "b1_results",
      "predictions": "b1_predictions"
    }
  }
}
```

Allowed artifact kinds are `csv`, `json`, `yaml`, `image`, `binary_mask`,
`probability_npy`, `probability_png`, `torch_checkpoint`, and
`joblib_model`. A method runner is one of:

- `deep_checkpoint`: a receipt-authenticated YAML/JSON config and PyTorch
  checkpoint, loaded with `weights_only=True`;
- `classical_joblib`: a receipt-authenticated fitted classical model, with
  optional `target_ppi` and `min_component_blocks`; or
- `precomputed_only`: available in frozen panels but not for uploads.

The test manifest columns are `example_key,label,image_id,examiner_value,ppi,
image_artifact,ground_truth_artifact,split`. The B1 columns are
`example_key,label,image_id,group,device,capture,ppi,image_artifact,
hull_artifact,minutiae_artifact,split`. Both must say `test` in every `split`
cell. Labels are private display labels; artifact columns are receipt keys, not
paths.

Prediction indexes contain `example_key,method,artifact,artifact_kind,
representation,block_size,min_component_blocks`. Representations are `pixel`,
`block`, or `mask`. Pixel and block entries use authenticated probability maps;
mask entries support only the reported threshold. Block entries use the frozen
16-pixel grid and shared post-processing.

Test result rows contain `example_key,method,dice,mdr,fdr,ms_per_img`. B1 rows
contain `example_key,method,mc,hr,par,ohf,ms_per_img`. The comparison view
recomputes Dice from the authenticated prediction and GT and refuses a stored
value that differs by more than the serialization tolerance.

## Environment and factory

Set these values in the private process environment (never a committed file):

| Variable | Meaning |
|---|---|
| `DEMO_RESTRICTED_ROOT` | Root containing receipt and all authenticated artifacts |
| `DEMO_ARTIFACT_RECEIPT` | Receipt path inside that root |
| `DEMO_ARTIFACT_RECEIPT_SHA256` | Independently pinned lowercase receipt digest |
| `DEMO_INFERENCE_DEVICE` | `cpu` or `cuda`; CPU is the default |
| `DEMO_CUDA_MEMORY_FRACTION` | Allocator fraction in `(0, 0.1]` |
| `DEMO_BACKEND_FACTORY` | `demo.restricted_backend:create_backend` |
| `DEMO_HOST` | Numeric loopback or private/VPN address only |
| `DEMO_PORT` | Private listening port |
| `DEMO_USER`, `DEMO_PASS` | Basic-auth credentials |
| `DEMO_REPORT_PATH` | Private report `index.html` with the builder marker |

Use the project launcher only after the receipt, report and credentials have
been provisioned. The code rejects public, wildcard and hostname binds;
`share=True` has no code path. Never pass an upload through a filesystem-backed
Gradio component.

## Operational checks

Before launch:

1. run the complete CPU test suite;
2. confirm the receipt digest by an independent command/channel;
3. confirm every method intended for Upload has a model runner;
4. check that all manifest rows are test-only and the B1 table uses hull metrics;
5. verify the report carries the private-report marker; and
6. run the repository privacy scan.

After launch, exercise all five tabs on synthetic fixtures first. The 1 GB
encoded-upload ceiling and 100-million-pixel decoded ceiling are enforced in
both the UI and backend. Uploads are decoded from an in-memory data URL and are
never persisted by application code.
