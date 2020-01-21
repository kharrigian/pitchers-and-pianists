# Pitchers and Pianists: Tapping Analysis

Standalone programs are housed within `scripts/`, with helper functions being relegated to `scripts/libraries/*`.

### Setup

This codebase assumes you are using python 3.6+. Primary dependencies can be found within `requirements.txt`. Nearly all packages can be installed using `pip`, except `peakutils` which can be installed from source from the directions here at https://peakutils.readthedocs.io/en/latest/.

### Signal Processing

The first task of this project was to identify tap initations in raw digital force signal. This ended up being non-trivial due to substantial external noise in the data. Issues were ultimately mitigated using a 4-stage approach.

1. `scripts/tap_processing_stage_1.py` - Manually look through all trials and remove subjects whose data will likely be unrecoverable.
2. `scripts/tap_processing_stage_2.py` - Use a Hidden Markov Model (HMM) and additional logic to identify tap initiations.
3. `scripts/tap_processing_stage_3.py` - Implements a simple GUI to manually check tap detections and remove false positives.
4. `scripts/tap_processing_stage_4.py` - Removes outlier Intertap Intervals based on deviation from mean.

### Experimental Cleaning

* `scripts/evaluate_preferred_period_calculations.py` - Runs tap detection algorithm on data used to calculate preferred period online and compares results to online calculation (which used a different algorithm)

### Data Consolidation

* `scripts/extract_tap_metrics.py` - Computes metrics like error, drift, and variability within appropriate signal windows.

### Statistical Analysis

1. `scripts/statistical_testing.py` - Uses Mixed Linear Models to evaluate fixed and random effects on dependent measures.
2. `scripts/intertask_analysis.py` - Within-subject comparisons beteen tapping task performance and skittles task performance.

### Miscellaneous

* `scripts/methodology.py` - Visualizations of the experimental methodology, in addition to comparisons of performance between two tap detection algorithms.
