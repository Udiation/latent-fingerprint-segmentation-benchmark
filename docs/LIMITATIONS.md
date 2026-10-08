# Limitations

- Several methods are faithful reimplementations because original training code
  or weights were unavailable. Differences from an original implementation can
  affect absolute performance.
- SegFormer-B2 is a declared substitute for an OCT-specific transformer, not a
  reproduction of that reviewed model.
- SUFS and FingerNet are released zero-shot models trained on external data;
  they do not share the benchmark training distribution with from-scratch rows.
- Examiner ROI and usable-ridge masks answer different segmentation questions.
  Reporting both exposes, but does not eliminate, sensitivity to GT definition.
- The SD302 exemplar minutiae hull is only a lower bound of fingerprint area.
  Foreground outside the hull cannot be classified as error.
- The principal data source is one coordinated collection. Subject-disjoint
  evaluation prevents identity leakage but does not establish generalisation to
  every operational sensor, population, substrate or examiner workflow.
- Timing comparisons report device class and include pre/post-processing, but
  remain sensitive to software stack and deployment details.

