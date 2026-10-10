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
  Foreground outside the hull cannot be classified as error, so pixel Dice is
  not reported for SD302b.
- The SD302b rolled table covers 430 test captures from 40 subjects, whereas
  the plain table contains only 9 captures from 5 subjects. Those 9 captures
  are segmented-slap impressions and do not establish performance for a broad
  conventional single-finger plain distribution.
- Predicted foreground area and inter-method pairwise Dice on SD302b are
  stability descriptors—agreement, not accuracy—because full pixel masks are
  unavailable.
- The aligned RA-RUnet row is unavailable because no authenticated aligned
  checkpoint completed before the benchmark freeze. Live-scan data were
  available, and the scored no-alignment RUnet is not substituted for it.
- The principal data source is one coordinated collection. Subject-disjoint
  evaluation prevents identity leakage but does not establish generalisation to
  every operational sensor, population, substrate or examiner workflow.
- Timing comparisons report device class and include pre/post-processing, but
  remain sensitive to software stack and deployment details.
