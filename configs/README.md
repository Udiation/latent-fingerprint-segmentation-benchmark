# Method configurations

These YAML files define the frozen architecture, data transformation and
search spaces. Fields labelled `pending` are populated by the release builder
from validation-only selection receipts after the seed-zero gate passes. No
test metric is used to choose a learning rate, threshold or stopping epoch.

The directory contains explicit configurations for every trainable or released
benchmark row. Classical grids (`svm`, `rf`, `welm`, and `gmfs`) and zero-shot
threshold selection (`sufs_zero`, `fingernet`) are configuration-controlled as
well; they are not hidden in report-generation code.
