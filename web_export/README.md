# Web export

Small JSON files for interactive figures on the research website. They are
generated, not edited:

```bash
python -m scripts.export_web_data      # or: make web
```

The export reads only `data/processed/` and `data/stimuli/`, is deterministic
(fixed order, fixed rounding, no timestamps) and totals about 400 kB. Every
file starts with `data_version`, `description` and `source_tables`.

## Shared conventions

- **Model ids** are the `id` values in `meta.json` (Hugging Face repository id
  with `/` replaced by `_`). Use `meta.json` for display names and
  architecture.
- **Margins** are normalised nearest-centroid margins: dimensionless, in
  [−1, 1], with 0 the decision boundary. Unless stated otherwise positive means
  closer to the intended (finally correct) sense. Rounded to 4 decimals.
- **Fractions and rates** are proportions in [0, 1], rounded to 4 decimals.
- **Layers** are hidden-state indices: 0 is the embedding output, `last_layer`
  the final layer. `relative_depth` = layer / `last_layer`.
- **GDV** is dimensionless; more negative means stronger sense separation.
  Rounded to 5 decimals.
- `null` means not defined (for example a rate with a zero denominator).

## `meta.json` (3 kB)

Source: `models.csv`, `stimuli/homonyms.csv`.

| Field | Content |
|---|---|
| `models[]` | `id`, `hf_repo_id`, `name` (display name), `architecture` (`encoder`/`decoder`), `n_hidden_states`, `last_layer`; in the fixed study order (4 encoders, then 4 decoders) |
| `words[]` | `word` and its two `senses[]`: `sense` (0/1), `label`, `short_label` |
| `margin_convention` | the sign convention as a sentence |

## `layer_separation.json` (72 kB) — model / homonym / layer selector

Source: `h1_layer_profiles.csv`, `h1_layer_selection_summary.csv`,
`q4_layer_curves.csv`.

`cells[model][word]` holds parallel arrays, one entry per layer:

| Field | Unit | Content |
|---|---|---|
| `layers` | index | hidden-state indices |
| `relative_depth` | 0–1 | layer / last layer |
| `fraction_adequate` | proportion | share of the 40 profiling sentences closer to their own sense centroid (leave-one-out) |
| `mean_margin_norm` | margin | mean leave-one-out normalised margin |
| `gdv` | GDV | Generalized Discrimination Value |
| `paired_context_fraction_adequate` | proportion | exploratory (Q4): adequacy of the 10 sentences whose resolving clause precedes the homonym |

and scalars: `selected_layer_adequacy`, `selected_layer_gdv`, `last_layer`
(layer indices), `nested_fraction_adequate_selected`,
`nested_fraction_adequate_last` (held-out proportions from nested
cross-validation).

## `revision_trajectories.json` (103 kB) — scrubbed sentence trajectories

Source: `h5_sentence_level.csv`, `stimuli/conflict_items.csv`.

| Field | Content |
|---|---|
| `stages` | `["prime", "homonym", "resolution"]`; the order of every per-stage array |
| `sentinel` | the readout token appended after each prefix |
| `items[item_id]` | `word`, `primed_sense`, `correct_sense`, `resolution_word`, `conflicting_sentence`, `coherent_control_sentence`, `stage_text[3]` (the text the model has seen at each stage, sentinel removed) |
| `trajectories[model][word].layer` | layer read |
| `…item_ids[]` | 14 item ids; the order of the arrays below |
| `…margins[][3]` | correct-sense margin at each stage, conflicting path |
| `…coherent_control_margin[]` | correct-sense margin at the resolver, coherent control path |
| `…resolver_only_margin[]` | correct-sense margin for the resolution word alone |
| `…primed_at_homonym[]` | boolean: margin after the homonym < 0 |
| `…resolved_correct[]` | boolean: margin after the resolver > 0 |

An item crossed the decision boundary when `primed_at_homonym` and
`resolved_correct` are both true.

## `trajectory_examples.json` (14 kB) — trajectories in a 2-D projection

Source: `example_states/*.npz`. Two examples (`bank` in
`FacebookAI_roberta-large` and `Qwen_Qwen2.5-7B`), each with two views:

- `revision` (H5): stages `prime`, `homonym`, `resolution`.
- `context` (H3): stages `homonym_position`, `sentence_final_position`, for
  items with the resolving clause before (`L`) or after (`R`) the homonym.

Per view: `layer`, `explained_variance_ratio[2]`, `xlim`, `ylim`,
`profile_points[][2]` with `profile_senses[]` (the background cloud of
profiling sentences), `sense_centroids[2][2]`, and `items[]` with
`points[][2]` (one per stage), `margins[]` (one per stage) and
`resolved_correct`.

Coordinates are principal-component scores of standardised hidden states
(arbitrary units, 3 decimals). They are a display projection that keeps a
minority of the variance: `margins` are computed in the full hidden space and
can disagree with apparent distances in the plane. Coordinates can differ in
the last decimals across linear-algebra libraries.

## `revision_comparison.json` (189 kB) — coherent versus conflicting

Source: `h5_sentence_level.csv`, `h5_by_layer_aggregate.csv`.

The same summary object is given at several levels: `overall`,
`by_architecture`, `by_model`, `by_word`, `by_direction` (`0_to_1`, `1_to_0`:
primed sense to correct sense) and `by_model_word` (which adds `layer`).

| Field | Unit | Content |
|---|---|---|
| `n` | count | model–item observations |
| `mean_margin_prime`, `mean_margin_homonym` | margin | conflicting path, before the resolver |
| `mean_margin_conflicting_resolved` | margin | conflicting path, after the resolver |
| `mean_margin_coherent_control` | margin | coherent path, after the resolver |
| `mean_margin_resolver_only` | margin | resolution word alone |
| `mean_movement` | margin difference | resolver minus homonym stage |
| `n_moved_toward_resolved`, `fraction_moved_toward_resolved` | count, proportion | observations with positive movement |
| `n_primed_at_homonym` | count | observations on the primed side after the homonym |
| `n_crossed_boundary`, `crossing_rate` | count, proportion | of those, the ones on the correct side after the resolver |
| `mean_conflict_minus_control` | margin difference | conflicting endpoint minus coherent endpoint |
| `mean_conflict_minus_resolver_only` | margin difference | conflicting endpoint minus resolver-only baseline |

`by_model_word_layer[model][word][]` is exploratory: a reduced summary
(`layer`, `n`, `mean_margin_homonym`, `mean_margin_conflicting_resolved`,
`mean_movement`, `n_primed_at_homonym`, `n_crossed_boundary`,
`mean_conflict_minus_control`) at every candidate layer of the Q4 extension,
for a layer scrubber. The number of layers differs between cells.

## `prior_heatmap.json` (18 kB) — isolated word versus ambiguous context

Source: `h0_summary.csv`, `h0_carrier_lean.csv`.

Sign convention here: **positive = closer to sense 0, negative = closer to
sense 1** (see `meta.json` for what the senses are).

| Field | Shape | Content |
|---|---|---|
| `models`, `words` | 8, 7 | row and column order of every matrix |
| `bare_word_lean` | 8 × 7 | signed margin of the homonym presented alone |
| `carrier_mean_lean` | 8 × 7 | mean signed margin over five ambiguous carrier sentences |
| `carrier_mean_abs_lean` | 8 × 7 | mean absolute margin over the carriers |
| `carrier_direction_consistency` | 8 × 7 | larger share of carriers leaning the same way (0.5–1) |
| `layer` | 8 × 7 | layer read |
| `carriers[model][word][]` | 5 | `carrier` (sentence) and its signed `lean` |
