# Web export

JSON files for interactive figures on the research website. They are
generated, not edited. There are two groups.

**Table exports** (six files, about 400 kB) are derived from the released
tables alone:

```bash
python -m scripts.export_web_data      # or: make web
```

**Display-geometry exports** (`geometry_by_layer.json`,
`garden_path_landscapes.json`, about 5 MB together) are projections of hidden states. The
hidden states are too large to release, so these two files can only be rebuilt
by someone who holds them (see REPRODUCIBILITY.md):

```bash
python -m scripts.build_geometry_exports --results-dir results --h5-matched-dir results/h5_matched
```

Both steps are deterministic (fixed order, fixed rounding, no timestamps): two
runs on the same inputs write byte-identical files. `make validate` checks all
eight files. Every file starts with `data_version`, `description` and
`source_tables`; the two display-geometry files also carry
`"status": "preliminary"`.

## Analysis values and display coordinates

Two kinds of numbers appear in these files and they must not be confused.

- **Full-space analysis values**: GDV, fraction adequate and margins. They are
  computed in the model's full hidden space (1,024 to 5,120 dimensions) and are
  the results of the study. In the display-geometry files their names end in
  `_full_space` or start with `full_space_`.
- **Display coordinates**: `points`, `profile_points`, centroids and density
  grids. They are a three-dimensional PCA projection made for plotting. In
  `geometry_by_layer.json` they are **within-layer standardised display
  coordinates in one shared PCA basis**. A
  projection keeps only part of the variance (the share is given as
  `explained_variance_ratio`), so distances on screen can disagree with the
  full-space values. A caption should say so and should take any number it
  quotes from the full-space fields.

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

## `geometry_by_layer.json` (2.2 MB) — sense geometry at every layer

Source: per-layer hidden states of the profiling sentences at the homonym
token (not released), `h1_layer_profiles.csv`, `stimuli/profiling_sentences.csv`.
Status: preliminary.

The coordinates are **within-layer standardised display coordinates in one
shared PCA basis**. They support inspecting how well the two sense clusters
separate at each depth. Absolute cloud size is not comparable between layers,
because every layer is standardised on its own. Full-space GDV, adequacy and
margins remain the analysis metrics.

For a layer slider next to the curves of `layer_separation.json`: the 40
sense-labelled profiling sentences of a homonym as a point cloud, at **every**
hidden-state layer of **every** model (56 model–homonym cells, 1,708 layers in
total).

`cells[model][word]`:

| Field | Content |
|---|---|
| `n_layers`, `last_layer` | number of hidden states; index of the final layer |
| `explained_variance_ratio[3]` | share of the pooled, standardised variance kept by each display axis |
| `source_activations_sha256` | hash of the hidden states the cell was projected from |
| `coordinates_sha256` | hash of the published coordinates of the cell |
| `layers[]` | one entry per layer, from 0 to `last_layer` |

`layers[]` entries:

| Field | Kind | Content |
|---|---|---|
| `layer`, `relative_depth` | index, 0–1 | hidden-state index; layer / last layer |
| `profile_points[40][3]` | display | coordinates of the 40 sentences |
| `profile_senses[40]` | label | sense (0/1) of each point |
| `sense_centroids[2][3]` | display | mean of the points of each sense |
| `n_profile_points` | count | 40 |
| `gdv_full_space` | full space | GDV of the layer, copied from `h1_layer_profiles.csv` |
| `fraction_adequate_full_space` | full space | leave-one-out fraction adequate, copied from `h1_layer_profiles.csv` |

Point `i` is the sentence with `profile_index` `i` in
`data/stimuli/profiling_sentences.csv`, at every layer, so a point can be
followed through depth and labelled with its sentence.

**Projection.** One basis per model and homonym, shared by all of its layers,
so moving the slider changes the geometry and not the coordinate system:

1. At each layer, subtract the midpoint of the two sense centroids (the senses
   are balanced, so this is the layer mean) and divide by the mean distance of
   the 40 states to their own sense centroid.
2. Pool the standardised states of all layers.
3. Fit a three-component PCA once (scikit-learn, `svd_solver="full"`; an exact
   decomposition, so no random state is involved).
4. Transform every layer with that basis. Coordinates are rounded to 4
   decimals.

Step 1 scales as well as centres. Without the scaling, layers with very large
activation norms (late decoder layers) would define the basis alone and every
other layer would collapse to a dot. The scaling is the project's existing
standardisation for pooled-layer projections
(`analysis/plot_centroid_separation.py`). One unit is the typical within-sense
spread of that layer, so the separation of the two clouds is comparable across
layers and their absolute size is not.

At layer 0 of models whose embedding output does not depend on context, the 40
states are identical and all points sit at the origin. That is the data, not a
gap.

## `garden_path_landscapes.json` (2.8 MB) — conflicting and coherent paths

Source: sentinel hidden states from `scripts/extract_h5_matched_states.py`
(not released), `stimuli/conflict_items.csv`, `h5_sentence_level.csv`,
`h5_aggregate.csv`. Status: preliminary.

For every model and homonym (56 cells), at the layer the H5 analysis used:
every context-conflict item (14 per cell, 784 in total) as a three-stage
trajectory, together with the trajectory of its matched coherent control, the
profile point cloud, and a density landscape of that cloud.

Stages are `prime`, `homonym`, `resolver`: the text before the homonym, through
the homonym, and through the resolution word. Each prefix is run on its own
with the sentinel appended and is read at the sentinel. `resolver` is the stage
called `resolution` in the other files.

`cells[model][word]`:

| Field | Kind | Content |
|---|---|---|
| `analysis_layer` | index | layer read; equal to `layer_used` in `h5_aggregate.csv` |
| `projection` | metadata | `method`, `dimensions` (3), `explained_variance_ratio[3]` (of the 40 profile states), `source_states_sha256`, `coordinates_sha256` |
| `profile.points[40][3]`, `profile.senses[40]`, `profile.centroids[2][3]` | display | sentinel states of the 40 profiling sentences, their senses, and the mean point of each sense |
| `n_items` | count | number of published item pairs |
| `default_item_id` | label | item to show first: the first item whose control lies on the same side of the boundary as in the released table |
| `control_resolver_max_abs_difference_from_released_table` | full space | see "Reproducibility discrepancy" |
| `items[]` | | one entry per item, see below |
| `unavailable_items[]` | | items without a matched control at all three stages (`item_id`, `reason`); empty in this release |
| `landscape` | display | density grid, see below |

`items[]` entries: `item_id`, `target_sense` (the sense the sentence resolves
to), `primed_sense`, `resolution_word`, `conflicting_sentence`,
`coherent_control_sentence`, and two paths, `conflicting` and
`coherent_control`, each with:

| Field | Kind | Content |
|---|---|---|
| `stage_names[3]` | label | `prime`, `homonym`, `resolver` |
| `stage_text[3]` | text | the text the model has read at each stage, sentinel removed |
| `points[3][3]` | display | position at each stage, in the same space as `profile.points` |
| `full_space_correct_margins[3]` | full space | normalised margin toward `target_sense` at each stage |

`coherent_control` has two more fields: `released_table_resolver_margin` (the
value of `matched_control_correct_margin_norm` in `h5_sentence_level.csv`) and
`resolver_side_differs_from_released_table`.

The two paths of an item are the real matched pair from
`data/stimuli/conflict_items.csv`: they share the text from the homonym onward
and differ in the opening context. A selector should offer one item at a time;
the file is not meant for drawing all items of a cell as a background.

**Projection.** Subtract the midpoint of the two profile sense centroids,
divide by the mean distance of the profile states to their own centroid, fit a
three-component PCA (`svd_solver="full"`) on the 40 profile states only, and
transform both paths with it. The trajectory states take no part in the fit.

**Density landscape.** `landscape` is a two-dimensional Gaussian kernel density
estimate of the profile points on the first two display axes, for a contour or
terrain view of the two sense regions.

| Field | Content |
|---|---|
| `x[33]`, `y[33]` | grid coordinates on display axes 1 and 2; the grid covers the profile points and all published paths |
| `sense_0_density`, `sense_1_density` | `[33][33]`, value at `(x[ix], y[iy])` stored as `density[iy][ix]`; each integrates to 1 over the plane |
| `total_density` | mean of the two |
| `bandwidth` | kernel standard deviation in display units: `20^(-1/6)` times the pooled within-sense standard deviation of the two plotted coordinates |
| `contour_levels[5]` | 0.1, 0.25, 0.5, 0.75 and 0.9 of the maximum of `total_density` |

The height of a terrain drawn from this grid is the local density of the 40
profile sentences in a two-dimensional projection. It is not the third
principal component, not a probability the model assigns, and not an energy or
any other model quantity. The third display axis is available separately in
every `points` entry.

**Reproducibility discrepancy.** Every path is one real path: its three
stages come from the same forward-pass group and are never combined from two.

- The conflicting paths repeat the inputs of the published H5 run and reproduce
  `h5_sentence_level.csv` to within 0.0001 at all three stages.
- The coherent-control paths were read in a separate group, because the
  published run read the control at the resolver stage only. Under the
  pipeline's bfloat16 arithmetic a state depends slightly on which texts share
  its batch, so the control's resolver margin is not identical to the released
  `matched_control_correct_margin_norm`: over the 784 items it differs by
  0.0005 on average and by at most 0.027, and one lies on the other
  side of the boundary. The statistics are in `numerical_note`; the released
  value and a side flag are stored with every control path.

An item with `resolver_side_differs_from_released_table: true` must not be
used as a default or showcase example; `default_item_id` never points to one.
In this release that is one item of 784.

The cause and its tests are documented in `STATUS.md` ("Batch dependence of
hidden states"). Differences between paths smaller than about 0.01 should not
be interpreted.
