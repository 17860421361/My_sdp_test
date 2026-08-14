# Fig. 1 — Motivation for configurable CSI preprocessing

- **Figure goal:** Establish preprocessing as a first-class experimental variable rather than an implementation detail.
- **Paper claim:** Raw CSI combines sensing-related dynamics with heterogeneous acquisition artifacts. Because every preprocessing operator imposes a signal prior, it can either preserve or attenuate task-relevant information; a fixed model can therefore yield different measured outcomes under different pipelines.
- **Figure type / mode:** Four-panel conceptual mechanism figure / image-mode brief, realized as editable SVG primitives.
- **Data:** Not applicable. The waveforms and bars are schematic and do not encode measured values.
- **Panels:**
  1. Mixed raw CSI: motion dynamics, phase drift, and impulsive noise coexist.
  2. Operator-specific priors: denoising, outlier repair, phase calibration, normalization, and interpolation make distinct assumptions.
  3. Information consequence: matched processing retains a highlighted fast motion signature; mismatched processing attenuates it.
  4. Evaluation consequence: with model and split fixed, pipelines A and B lead to qualitatively different measured outcomes.
- **Style contract:** IEEE two-column-width figure; white background; four aligned panels; Helvetica/Arial; blue for the central method path; green for preserved information; red for attenuated information; charcoal/gray for context; no gradients, shadows, fictitious numerical values, or dense explanatory prose.
- **Output:** `fig1_csi_preprocessing_motivation.svg` (editable text, paths, lines, and shapes).
- **Verification:** labels remain readable at `\textwidth`; reading order is left-to-right; every colored path has a consistent meaning; phase calibration is shown only as conditional on complex CSI; no empirical metric is implied.
