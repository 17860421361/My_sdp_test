# Fig. 2 brief — Preprocessing-centric CSI evaluation framework

- **Figure goal:** Explain how heterogeneous CSI data are standardized through SDP, processed by a data-availability-aware configurable pipeline, and evaluated through a two-stage experimental design.
- **Paper claim:** Preprocessing is an explicit, configurable experimental factor. Its applicability depends on retained signal information, and its effects are assessed through cross-model screening, fine-grained combinations, and evidence-based diagnosis.
- **Figure type / mode:** System architecture / experimental workflow; vector-native SVG.
- **Narrative:** Four datasets → SDP reader and parser → CSI representation plus labels → signal-availability check → valid plug-in operators in a fixed execution order → model-ready input → shared experimental protocol → Stage I preset-by-model screening → representative-model selection → Stage II fine-grained valid combinations → accuracy, waveform, and ablation evidence → effects and applicability conditions.
- **Required labels:** Widar3.0, GaitID, XRF55, ElderAL-CSI, SDP unified reader & parser, Denoising, Outlier handling, Phase calibration, Normalization, Interpolation, Stage I, Stage II, Accuracy, CSI waveform changes, Component ablation.
- **Style:** IEEE/Visio-like, English only, white background, editable native SVG, restrained blue/green/gray semantics, no fabricated values.
- **Output:** `fig2_preprocessing_centric_evaluation_framework.svg`.
- **Verification:** data flow is explicit; stage I and stage II are experimental phases rather than signal-processing steps; phase calibration is marked applicable only when reliable complex CSI is retained; no numerical result is implied.
