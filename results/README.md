# Public aggregate results

`public_results.csv` is the only metric source consumed by the public website.
Rows remain `pending` with empty metrics until the seed-zero completion gate.
Final imports must be aggregate by method and contain no sample identifier.

`table_status.csv` controls which tabs can be rendered. `statistics.csv` holds
aggregate confidence intervals and corrected tests only. The build and number
checker reject unapproved columns and non-finite or out-of-range metrics.

