# Synthetic examples

Everything in this directory is generated and hash-bound by
`synthetic_receipt.json`. The images are pipeline fixtures, not biometric
samples and not benchmark evidence. `quickstart.ipynb` runs the classical smoke
path without saved outputs.

The project-page demo always exposes the generated ground-truth mask. A model
output is added only after that implementation has actually run on the same
synthetic source. Its receipt entry must have:

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
digests, and receipt/file mismatches. It never substitutes or fabricates a mask
for a method that has not produced one. A complete public demo requires a
receipt-bound prediction from every applicable benchmark implementation for
each displayed source; until then, the method selector honestly remains
partial.
