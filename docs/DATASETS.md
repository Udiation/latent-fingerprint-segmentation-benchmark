# Datasets and ground truth

No biometric sample, annotation, mask or per-image identifier is distributed
with this repository.

## Requesting NIST SD302

Use the [official NIST Special Database 302 page](https://www.nist.gov/itl/iad/btg/nist-special-database-302)
and follow NIST's current access and usage instructions. The benchmark consumes:

- **SD302e:** latent distal-phalanx source images;
- **SD302h:** EFS examiner annotations corresponding to those latent images;
- **SD302g:** annotated exemplar transactions for the lower-bound live-scan
  analysis;
- **SD302b:** baseline live-scan captures matched exactly to their canonical
  SD302g records for lower-bound hull evaluation and available as unlabelled
  alignment inputs;
- **SD302c:** only when a protocol explicitly needs additional unlabelled
  alignment inputs or independently human-annotated masks.

## GT reconstruction

`scripts/build_sd302h_gt.py` parses traditional ANSI/NIST records, rasterises
field 9.300 examiner polygons and separately rasterises the field 9.308 ridge
quality grid. The parser uses each record's declared dimensions and resolution;
it never adjusts coordinates to improve visual alignment.

`scripts/build_sd302g_hulls.py` parses expert minutiae in field 9.331 and forms
a convex hull after physical-unit conversion. Field 9.300 in that partition is
not interpreted as a pixel foreground outline. The generated hull is restricted
to lower-bound evaluation and must never be used as supervised foreground GT.
An SD302g exemplar is reused for SD302b scoring only when its embedded image and
the canonical SD302b image have identical dimensions and pixels.

## Counts and exclusions

- 10,092 annotated latent impressions from 200 subjects;
- 10,024 complete ridge-quality maps;
- 68 quality-map-incomplete records excluded from A-prime only;
- two NONPRINT annotations excluded from final test scoring;
- frozen subject counts: 136 train, 24 validation, 40 test;
- 430 exactly mapped rolled SD302b test captures from 40 subjects;
- 9 exactly mapped plain SD302b test captures from 5 subjects, all with
  segmented-slap capture semantics.

Source images with multiple annotation records are aggregated to one image-level
study unit before model fitting. The public results state image-level counts for
each endpoint after this aggregation.

No full pixel masks were available for these SD302b subsets. Their tables use
the minutiae hull as a lower bound and therefore report MC, HR, PAR and OHF, not
pixel Dice. Predicted-area fraction and inter-method pairwise Dice are labelled
agreement, not accuracy.

## Filename explanation for privacy scanning

Original NIST filenames can encode an eight-digit anonymous subject number,
resolution tokens such as `1000PPI`, and latent-position tokens such as `LP01`.
These patterns are documented here only so the pre-push scanner can reject
their accidental appearance everywhere else in the repository.
