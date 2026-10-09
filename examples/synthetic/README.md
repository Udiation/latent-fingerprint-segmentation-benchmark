# Synthetic examples

Everything in this directory is generated and hash-bound by
`synthetic_receipt.json`. The images are pipeline fixtures, not biometric
samples and not benchmark evidence. `quickstart.ipynb` runs the classical smoke
path without saved outputs.

The project-page demo always exposes the generated ground-truth mask. Run a
core-only CI-sized build with:

```bash
python scripts/build_synthetic_site_demo.py --methods rf --steps 1 --threads 1
```

After installing the optional deep and pretrained requirements, the slower
artifact build is:

```bash
CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=torch \
  python scripts/build_synthetic_site_demo.py --methods all --steps 30 --threads 2
```

The builder fits trainable methods on declared synthetic train subjects,
selects thresholds on a separate synthetic validation subject, and runs each
completed implementation on the same three displayed sources. It records a
content digest for the builder, manifest, implementation sources, method
record and prediction set. A model output is added only after that code path
has actually run. Its receipt entry includes:

```json
{
  "path": "predictions/<method>/<synthetic-name>.png",
  "sha256": "<digest>",
  "mime_type": "image/png",
  "role": "method_prediction",
  "method": "<stable_method_key>",
  "label": "<display label>",
  "source_stem": "<synthetic-name>"
}
```

The site builder rejects path traversal, malformed method keys, missing
digests, incomplete three-source output sets, source/record mutations, and
receipt/file mismatches. It never substitutes or fabricates a mask for a
method that has not produced one. FingerNet and the ImageNet-initialized
SegFormer substitute need pinned external snapshots that the public repository
does not distribute, so they are explicit `unavailable` entries. A limited
build marks other unrequested rows `not_built`.

These micro-runs only demonstrate that repository implementations execute on
generated patterns. Their masks, training losses and validation choices are
not benchmark measurements and must not be used to compare algorithms.
