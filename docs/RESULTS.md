# Results

> **Pending:** final aggregate metrics are not published until every Fig. 2
> entry has a seed-zero test result or an exact N/A reason.

The release builder will populate this page from the same aggregate CSV files
that drive the paper tables and project website. It will include:

- Table A (examiner ROI GT), A-value and A-prime (usable-ridge GT);
- Table B1 exemplar lower-bound metrics and device breakdowns;
- cross-domain Table D;
- subject-cluster bootstrap confidence intervals;
- Holm-corrected paired tests;
- Dice/runtime and examiner-value figures.

No metric is hand-transcribed into the public release. Run
`python scripts/check_site_numbers.py` after every result import.

