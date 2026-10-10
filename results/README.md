# Public aggregate results

`public_results.csv` contains the canonical Stage-5 aggregate metrics. The
v1.1 site additionally consumes the receipt-bound aggregate-only SD302b
supplement in `sd302b_results.csv`; rows are merged by endpoint, view, group,
and method, so the supplement may amend an explicitly documented unavailable
reason without altering the canonical Stage-5 evidence. No missing test metric
is inferred, and no public result file contains a sample identifier.

`table_status.csv` and `sd302b_table_status.csv` control which tabs can be
rendered. `statistics.csv` holds aggregate confidence intervals and corrected
tests only. `sd302b_release_receipt.json` binds both supplement files to the
frozen aggregation receipt and declares that they contain neither per-image
rows nor biometric media. The build and number checker reject unapproved
columns and non-finite or out-of-range metrics.

## Fail-closed import

Public results are updated only through `scripts/update_from_aggregate.py`.
The source directory must contain exactly:

- `public_results.csv` with the public aggregate result schema;
- `statistics.csv` with aggregate confidence intervals and corrected tests;
- `table_status.csv` with public table availability; and
- `aggregate_manifest.json`, whose exact schema is:

```json
{
  "schema_version": 1,
  "kind": "stage4_public_aggregate_bundle",
  "files": {
    "public_results.csv": "<lowercase SHA-256>",
    "statistics.csv": "<lowercase SHA-256>",
    "table_status.csv": "<lowercase SHA-256>"
  }
}
```

The importer rejects extra files, symlinks, changed hashes, unexpected or
per-image columns, biometric-style identifiers, embedded media, model/data
artifacts, private absolute paths, hosts, IP addresses, and cloud-drive
references. It stages and number-checks the marker-bound README headline table,
`docs/RESULTS.md`, and the project site before applying any output.

Always review a non-mutating receipt first:

```bash
python scripts/update_from_aggregate.py \
  --bundle /path/to/public-aggregate-bundle \
  --dry-run \
  --report /path/to/dry-run-report.json
```

After review, replace `--dry-run` with `--apply`. The report records only
schemas, counts, hashes, checks, and generated-output hashes; it never records
the source path.

## Canonical Stage-5 bridge

The final benchmark pipeline exports a richer, aggregate-only Stage-5 package.
`scripts/import_stage5_public.py` is the publication bridge from that canonical
schema to the compact CSVs above, the paper tables, aggregate figures, website,
chosen configs, and anonymous subject split. Its trust root is an independently
supplied SHA-256 of `package_receipt.json`; deriving the expected digest from the
same package is intentionally unsupported.

Alongside tables, statistics, captions, figures, and
`public/release_universe.json`, final release mode requires receipt-bound files
at the following exact extension paths:

- `public/configs/*.yaml` for selected public configurations; and
- `public/splits/sd302_subject_split.json` for the 136/24/40 anonymous split.

If those extensions are absent, a normal dry run reports them as absent and
does not invent or overwrite either artifact. `--require-release-metadata`
turns their absence into a hard failure and is mandatory for a final release.
See [PUBLICATION.md](../docs/PUBLICATION.md) for the controlled-apply and scan
phases.

The bridge also enforces the preregistered row universe rather than accepting
an arbitrary non-empty table: all 16 Dataset-1 method rows plus the two
references must be present for A, A′, and each of VALUE/LIMITED/NOVALUE; B1
requires the aligned RA-RUnet overall placeholder and hull reference. The
unavailable aligned row need not be repeated in device/capture breakdowns
because no prediction exists to stratify. D requires the
13 methods trained on SD302-Latent and evaluated on exemplars. Every observed
B1 group/device/capture bucket must contain that complete applicable method
set. The receipt also binds `public/release_universe.json`, whose exact
table/view/group/method projection must equal `tables.csv`. Unavailable methods
remain explicit rows with reasons. Across-seed standard deviations are carried
through the compact CSV, README, generated Markdown, and website; they are not
discarded during conversion.

Trainable pixel/patch configs declare `selection.status: selected`, the exact
seven-value validation grid, and a `threshold_by_seed` mapping whose keys equal
the canonical completed seed IDs. The two unseeded pretrained selectors use a
single `chosen_threshold`. A selector may be `unavailable` only when its reason
exactly matches every applicable unavailable table row; the bridge never fills
in a missing threshold. Classical block configurations similarly retain a
`chosen_by_seed` parameter mapping keyed by the canonical completed seeds; a
single `chosen` mapping is accepted only when it equals every per-seed choice.

The current producer can encode an unavailable point-estimate statistic with a
blank status and placeholder interval. The bridge normalizes only the exact
case where the same receipt-bound table/method row is explicitly unavailable:
it copies that reason and removes the placeholder statistics. The count is
reported as `canonical_reconciliations.unavailable_point_estimates`; every
other blank or unknown status fails closed.
