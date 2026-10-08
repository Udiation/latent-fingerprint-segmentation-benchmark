# Public aggregate results

`public_results.csv` is the only metric source consumed by the public website.
Rows remain `pending` with empty metrics until the seed-zero completion gate.
Final imports must be aggregate by method and contain no sample identifier.

`table_status.csv` controls which tabs can be rendered. `statistics.csv` holds
aggregate confidence intervals and corrected tests only. The build and number
checker reject unapproved columns and non-finite or out-of-range metrics.

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
references. It stages and number-checks `docs/RESULTS.md` and the project site
before applying any output.

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
