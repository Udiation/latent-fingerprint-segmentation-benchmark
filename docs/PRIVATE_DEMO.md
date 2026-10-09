# Private review demo contract

The private Gradio application is a review surface, not a second evaluation
pipeline. Its reported metrics must come from frozen predictions and the same
validation-selected thresholds used in the benchmark tables.

## Restricted boundary

`demo/backend.py` defines an interface only. A deployment-specific backend is
loaded from an environment-provided factory and remains outside the codebase.
That backend owns all dataset and model locations, private sample identifiers,
checkpoints, probability maps, timing records, and hull annotations. The UI
receives only in-memory arrays and display-safe labels.

The backend must provide:

- unique method names and each method's validation-selected threshold;
- private test examples with an examiner-value label, source image, PPI, and
  optional full-mask ground truth;
- thresholded predictions from the frozen inference path, including the
  threshold actually applied and the measured runtime;
- precomputed per-example Dice pairs for the two-method review; and
- live-scan examples with expert minutiae and their convex hull.

The production implementation is `demo.restricted_backend`. It accepts only an
environment-selected restricted root and an independently SHA-256-pinned
receipt. Manifests, result rows, probability maps, binary masks, checkpoints
and fitted classical models are each opened only after their byte count and
digest match the receipt. See `PRIVATE_DEMO_RUNBOOK.md` for the schema and
deployment procedure.

The UI checks shapes, numeric ranges, thresholds, runtimes, and keys before it
renders a result. A backend that does not implement the complete protocol is
rejected before launch.

## Metric separation

The **Explore test set** tab runs each chosen method twice: once with the
reported validation-selected threshold, and once with the reviewer's slider
value. Only the first result is scored. The slider output is labelled
"exploration only" and cannot alter the table row.

The **Compare two methods** tab uses reported-threshold masks. Its difference
map colours pixels where only method A is correct separately from pixels where
only method B is correct. The top-20 list is a display of frozen test results;
it is not a tuning or selection input.

The **Live-scan (SD302g)** tab treats the expert-minutiae convex hull as a lower
bound, never as complete pixel ground truth. It reports MC, HR, PAR, and OHF.
Dice, Jaccard, MDR, and FDR are deliberately absent.

## Upload and network controls

Uploads are read by the browser as a data URL, decoded to an array in memory,
and passed directly to frozen inference. The app has no Gradio file-backed
media component and does not write the input. Uploaded images have no ground
truth, so the tab reports only method thresholds and runtime with the overlays.

At launch, the app accepts only a loopback or private IP literal, requires
basic authentication, disables public sharing, monitoring, analytics, and
public API visibility, and blocks filesystem serving. The private report is
accepted only when it carries the report builder's private marker, and it is
embedded in a sandboxed frame.

The upload boundary rejects an encoded payload above 1 GB and checks decoded
dimensions before allocating the grayscale array; the decoded ceiling is 100
million pixels. Methods without an authenticated runnable model are omitted
from the Upload tab but remain available for authenticated frozen panels.

## Synthetic verification

`tests/test_demo_app.py` exercises all five workflows with generated arrays.
It verifies reported-versus-exploration threshold separation, difference-map
and top-20 behavior, exact reference hull metrics, in-memory upload behavior,
absence of file-backed Gradio media components, and enforced private launch
arguments. No biometric image, private identifier, model weight, or result is
used by the test suite.
