#  Where Do Semantics Live? A Homonym-Based Study of Semantic Disambiguation in Large Language Models

A reproducible study of whether transformer hidden states carry word-sense information in a way that can be read out across layers, token positions, and context conditions.

## Project report

The full write-up is [Where Do Semantics Live - Project Summary.pdf](Where%20Do%20Semantics%20Live%20-%20Project%20Summary.pdf), covering the hypothesis structure, methods, and results for all six analyses (H0–H5).

## How the hypotheses relate

H1 identifies candidate semantic layer(s); H0, H2–H5 each test, revise, or correct that result rather than standing alone. Tests increase in specificity — if H1 identifies a set of layers, H2–H5 help narrow them to fewer candidates, and if one layer dominates across all tests, it may contain a robust representation of contextual semantic processing.

| | Question | Role |
|---|---|---|
| **H0** — Starting bias | A model may be biased toward one sense of a homonym because of its training data. If a model consistently chooses one sense, that choice is not meaningful evidence in the later hypotheses. | Corrects results from H1–H5 |
| **H1** — Meaning across depth | Is language processing hierarchical in the model? If early layers mainly preserve word-form information, sense groups should overlap for both homonyms; if contextual meaning emerges with depth, they should increasingly separate. If a hierarchy emerges, there should be a single layer or region where semantic processing dominates. | Identifies semantic layer(s), if any |
| **H2** — Readout validation | If candidate semantic layer(s) can be identified, their role must be validated by testing whether they also classify unseen sentences and unseen homonyms. | Tests H1 result |
| **H3** — Context and anchor | If semantic layer(s) exist, token representations should show the model moving from an ambiguous homonym state toward either sense once resolving context is added — and this should differ between encoders and decoders, since they have different access to context. | Tests H1 result |
| **H4** — Decoder-anchor revision | Because causal decoders cannot access subsequent tokens, are the estimated starting biases affected by the homonym's position in the sentence? H1 is repeated at the end-of-sentence token rather than the homonym position to test whether the identified semantic layer(s) change. | Revises H1 result |
| **H5** — Meaning revision | H3 tests ambiguity resolution; H5 tests whether a model can revise an interpretation encouraged by earlier context when later context requires the alternative sense. | Tests H1 result |

### Methods per hypothesis

- **H0**: activation collection (bare word vs. carrier), centroid probing (two-sense geometry), sense-margin scoring (prior direction), carrier consistency (baseline stability)
- **H1**: layer-wise activations, centroid separation (sense distance), PCA trajectories (geometric progression), candidate-layer selection (semantic region)
- **H2**: nested leave-one-out (sentence generalization), held-out homonym test (homonym generalisation), adequacy (item-level accuracy), GDV (global class separation), layer comparison
- **H3**: resolver reordering (left- vs. right-context), PCA trajectories (ambiguity to resolution), correct-sense margin (movement direction), paired bootstrap (order effect)
- **H4**: position-specific centroids (endpoint vs. homonym position sense-geometry)
- **H5**: garden-path priming, fixed-sentinel probing (constant readout for controls), trajectories (prime to resolver), matched controls, boundary crossing

## Key results by hypothesis

- **H0 (starting bias)** — Priors exist for all models and homonyms, but they differ between isolated bare words and words presented in context.
- **H1 (meaning across depth)** — A clear depth-wise hierarchy emerges: in the geometric representation of token-level activations, the two senses begin in similar positions and diverge into sense clusters as context is added. However, semantic/contextual information dominates token-level activations quite quickly and, in some cases, plateaus. There is no single semantic layer, although some models may have broader semantic regions — for most models these appear between mid-depth and three-quarter depth.
- **H2 (readout validation)** — Candidate layers, including the selected winners, generalise. However, different metrics produce different winners and, in some cases, different candidate regions.
- **H3 (context and anchor)** — For decoders, the geometry shows a clear transition from an ambiguous shared location to distinct sense regions once the resolver is added. For encoders, left and right resolution make little difference: the geometric evolution suggests that sentence-level ambiguity moves token-level representations closer together regardless of resolver order, consistent with encoders having access to all tokens.
- **H4 (decoder-anchor revision)** — For decoders, H4 produces the same candidate regions and winning layers as H1. The short homonym-in-context sentences therefore appear sufficiently balanced not to skew candidate generation.
- **H5 (meaning revision)** — H5 consistently shows geometric movement from one sense towards the other, although revision does not always cross the decision boundary. The primed sense persists with different strength depending on the homonym and model, suggesting a substantial cost of revision rather than a clean overwrite.

## What this repo does

- studies seven homonyms: bank, bark, bat, crane, spring, match, pitch
- excludes light because its two senses differ in part of speech and confound semantic vs. syntactic-category resolution
- evaluates hidden-state sense evidence with a signed centroid margin $M_l$ and its normalized form $M_l^{norm}$
- runs the H0–H5 analyses and regenerates the report figures and CSV outputs

## Key folders

- data/: stimuli and synthetic profiling data
- hypotheses/: one runner per hypothesis (H0–H5)
- experiments/: margin, adequacy, and GDV computation
- models/: model loading and activation extraction
- results/: generated outputs (gitignored)

## Quick start

```bash
cd llm_homonym_semantics
python run_study.py --hypotheses H1 H2
python recompute_geometry.py
python plot_geometry_audit.py
python plot_semantic_layer_atlas.py
python plot_prior_matrix.py
python plot_h3_h4_audit.py
python plot_report_h1_h5.py
```

For the full study on an HPC cluster, a Slurm launcher can be used locally, but it is intentionally not part of the public repository baseline.

## Reproduction checklist

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full step-by-step workflow.

1. Ensure the offline Hugging Face cache is present under the workspace hf_cache directory.
2. Use the project Python environment that has the required dependencies installed.
3. Confirm the stimulus data files exist in data/.
4. Run the study or the full Slurm job.
5. Regenerate the figures and validate the outputs with validate_study_run.py.
6. Inspect the CSVs and plots under results/study/ and results/study/figures/.

## Notes

- Results, figures, and run-specific artifacts are generated under results/ and are not tracked by git.
- The current public-facing analysis uses the seven-word study set and excludes light.
- The repository is intentionally focused on source code, stimuli, and reproducibility rather than HPC launchers or report-specific plots.
