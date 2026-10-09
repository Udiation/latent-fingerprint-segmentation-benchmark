# FAQ

## Where are the fingerprint images and masks?

They are not distributed. Request SD302 from NIST and build the derived GT
locally by following [DATASETS.md](DATASETS.md). Repository images are synthetic
or aggregate plots only.

## Can I get the trained weights?

Weights may be available on request, subject to NIST SD302 terms and an
independent review. They are never attached to the public repository or site.

## Why are some result cells marked pending?

The release is fail-closed: partial test values are not promoted to headline
results before the preregistered seed-zero completion gate passes.

## Are the browser-demo masks benchmark results?

No. They are deterministic CPU micro-runs on generated ridge patterns. Their
receipt proves which implementation produced each mask, but their abbreviated
training budget is only an executable-code demonstration. Methods requiring
external snapshots that are not distributed here are shown as unavailable;
the site never fabricates a replacement output.

## How do I add a method?

Add a versioned YAML config, make inference emit a full-resolution probability
map at 500 ppi, select its threshold on validation only, and route its binary
mask through `scripts/evaluate_masks.py`. Add synthetic tests and document its
training provenance and any departure from the cited source.

## Why is there no Dice score for exemplar hulls?

The hull encloses expert minutiae but does not describe the full fingerprint
boundary. Dice would penalise valid foreground outside the hull. The benchmark
therefore reports MC, HR, PAR and OHF.
