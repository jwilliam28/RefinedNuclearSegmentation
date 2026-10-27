# RefinedNuclearSegmentation (`rns`)

This tool finds nuclei on brightfield images using two published detectors, StarDist and InstanSeg, run inside QuPath. It then redraws each outline so that it follows the nuclear membrane, and gives every nucleus a focus flag. It was built for Papanicolaou-stained cytology, where models trained on H&E find the nuclei but draw loose outlines. On H&E, run it with `--no-papmap`.

## What it fixes

| What goes wrong with H&E models on Pap | What `rns` does |
|---|---|
| The cyan cytoplasm rim is counted as part of the nucleus | **Colour remap:** the models see (max(R,G), G, B), so cyan cytoplasm looks bright. **Outline:** the boundary is placed where a per-image colour classifier marks the edge of the nucleus. |
| Pale parts of a nucleus are cut off | **Classifier training:** it learns from the inner half of *every* detected nucleus, pale ones included. **Boundary search:** rays move inward from outside the nucleus and stop at the first sustained nucleus pixels. |
| The detector outline stops short of a large or irregular nucleus | **Growth (optional, off by default):** with `grow_um > 0` the search area is extended along the nucleus probability, only through pixels at least 0.6× as dark as the nucleus core and only where the new part is closed off by cytoplasm. On Pap slides a cytopathologist's review of random cases found it changed outlines little and sometimes added cytoplasm, so the default is off. |
| A nucleus that touches similar nuclei is missed | **Lower threshold:** StarDist runs at threshold 0.35 with a 0.35 µm model pixel. **Second detector:** InstanSeg nuclei are added wherever StarDist found nothing. |
| One nucleus is split into pieces (often along a fold) | **Coarse check:** a coarse StarDist pass (threshold 0.5, 0.5 µm) can consolidate the pieces when no pale gap separates them. **Merge (optional, off by default):** with `merge_pieces` on, two touching pieces are joined when the union is compact and the chromatin continues across the cut; on Pap slides it mostly joined pieces of blurred clumps and cytoplasm, so it is off by default. |
| Outlines are jagged or follow chromatin clumps | **Fourier smoothing:** detail at a 2 µm wavelength is halved and finer detail is strongly damped, at the same physical scale on every scanner. **Outline choice:** the darkness outline (nuclear OD above an Otsu threshold) is used first; its threshold is estimated on the nucleus plus a surround ring scaled to the nucleus (glass left out), so it falls between nucleus and cytoplasm: it leaves out the cytoplasm halos that a softer edge invites, and does not split large nuclei into darker and paler chromatin. The radial outline is the fallback. |
| Folds, strands, specks, cytoplasm edges and *Candida* are outlined as nuclei, especially on slides with few or pale nuclei | **Nucleus filter:** a logistic model checks every object, using whether all three detectors found it, its contrast against the surroundings, chromatin texture, colour, size, shape and the slide's own nuclear signal. The packaged model was trained on cytopathologist-graded objects; you can retrain it on your own grades. |
| Outlines are drawn on defocused parts of thick groups | **Focus flag:** every nucleus is marked sharp, intermediate or blurred relative to the sharp nuclei in its own field, so analyses can include or exclude nuclei by focus. |

## How it works

1. **Detect (QuPath, headless).** `detect_boxes.groovy` runs the following in square boxes around points you supply:
   - StarDist `he_heavy_augment`, once at threshold 0.35 with a 0.35 µm model pixel and once at threshold 0.5 with a 0.5 µm model pixel;
   - optionally InstanSeg `brightfield_nuclei`.

   It writes the detections as GeoJSON in full-resolution image coordinates and saves each box as a PNG. Outputs left by an earlier run in the same folder are deleted first, and `boxes.json` records whether InstanSeg ran.
2. **Union.** The two detectors' nuclei are combined. An InstanSeg nucleus is added only where StarDist covers less than 10 % of it.
3. **Consolidate.** Fragments are replaced by one coarse-pass nucleus when that nucleus is compact (solidity ≥ 0.90) and covers each fragment, and either:
   - one fragment dominates (≥ 55 % of the coarse area); or
   - the coarse outline is not elongated (axis ratio ≤ 1.4), and the material between the fragments is as nuclear as the fragments: nuclear OD ≥ 0.85× the paler fragment and nucleus probability > 0.5.

   The second path catches a nucleus cut along a fold or groove, which is darker, not paler. Two touching nuclei meet along a paler line or sit side by side in an elongated outline, so they stay separate.
4. **Classify pixels.** A two-class logistic regression gives each pixel a nucleus probability. The negative class pools cytoplasm and glass. It uses:
   - optical density (OD) per channel and OD chromaticity;
   - OD sum;
   - Lab a\* and b\*;
   - nuclear OD;
   - local texture.

   It is trained on each image's own detections, so no manual labels are needed. Positives come from the inner half of each nucleus. Negatives come from a ring 1.5–4 µm outside it and from the brightest (glass) pixels; sampling the two separately makes sure both are represented.
5. **Find the boundary.** The search area for each nucleus is the detector outline plus 1.5 µm (with the optional growth, see the table, it can extend further), never into another detected nucleus. 128 rays then move inward from the edge of the search area:
   - each ray stops at the first stretch of at least 0.7 µm where the nucleus probability is above 0.5;
   - it then snaps to the steepest rise of nuclear OD within ±1 µm;
   - missing rays are interpolated, and the radii get a 5-ray circular median and a light Gaussian.

   An Otsu outline on nuclear OD is the alternative. It is used only when its mean edge gradient is at least 15 % higher. Each nucleus is processed in a small window around itself.
6. **Clean up.** Candidates are dropped if any of these holds:
   - area below 8 µm²;
   - mean nucleus probability below 0.4;
   - solidity below 0.75;
   - OD less than 1.15 times the nucleus's own 1–2.5 µm surround (glass and neighbouring nuclei are excluded from the surround).

   With `merge_pieces` on (off by default), two touching pieces are merged as halves (union solidity ≥ 0.93, elongation ≤ 1.7, area ≤ 1.6 times the median) or as a fragment (one piece ≥ 70 % of the union, solidity ≥ 0.90, and no pale line along the cut). Overlapping pixels go to the nucleus with the nearest centroid. The result is sub-pixel contours.
7. **Filter non-nuclei.** For every object, `nucleus_filter.py` measures:
   - **(A) detector agreement:** whether StarDist (fine), StarDist (coarse) and InstanSeg each have an outline containing it;
   - **(B) the object:** contrast against a 1–3 µm ring, chromatin texture (OD standard deviation and band-pass energy), OD chromaticity and Lab colour, size (log area, and how far below a typical nucleus it is, so specks and yeast score low without penalising large nuclei), solidity, circularity and elongation, and edge focus;
   - **(C) slide signal:** the image's median nuclear OD, and the object's darkness relative to it (computed and written to the table; the packaged model does not use it).

   Standardised logistic models turn these into P(nucleus): one for objects under 40 µm² and one for larger objects, so large dysplastic nuclei are judged on different evidence from small ones. Objects below the model's threshold are removed. The default threshold is strict: the smallest at which about 94 % of kept objects were nuclei on each of the two scanners, on slides the model had not seen. `run_info.json` counts the removed objects, and `nucleus_filter_table.csv` lists every object with its features.
8. **Smooth.** Each contour is resampled by arc length and its Fourier harmonics are damped with a Gaussian taper:
   - a 2 µm wavelength is halved, and about 6 % of a 1 µm wavelength is left;
   - the cutoff is set from the convex-hull perimeter, so jagged edges do not raise it;
   - folds and notches wider than about 3 µm survive.
9. **Flag focus.** The focus measure (FM) is the 90th percentile of the gradient of glass-normalised luminance, pre-smoothed at 0.5 µm, in a ±1 µm band around the outline. It is divided by the field's reference:
   - the reference is the 75th percentile of FM over the field's nuclei when the field has at least 5, otherwise the image-level value;
   - it is floored at 0.60 times the image's median field reference, so a field whose whole plane is out of focus is not normalised away;
   - FM / reference ≥ 0.75 is `sharp`, below 0.50 is `blurred`, and in between is `intermediate`.

Nuclear OD is min(OD_red, OD_green). Nuclei absorb both red and green, cyan cytoplasm absorbs only red, and pink cytoplasm absorbs only green, so the minimum is high only in nuclei.

Most refinement settings are fields of `Params` in [`src/rns/params.py`](src/rns/params.py), given in micrometres; override them with `rns refine --params my.json`. The StarDist thresholds and model pixel sizes are flags of `rns detect` and `rns run`. A few internal constants are fixed in `src/rns/refine.py`:
- the 0.5 probability cut;
- the 5-ray median;
- the 1–2.5 µm surround ring;
- the 0.08 OD glass cut.

## Install

```bash
pip install -e .            # or: pip install -e .[test]
python scripts/download_models.py          # StarDist .pb and InstanSeg brightfield_nuclei into ./models (SHA-256 checked)
```

You also need:
- **QuPath 0.6 or 0.7** ([qupath.github.io](https://qupath.github.io)).
- **The StarDist extension** ([qupath-extension-stardist](https://github.com/qupath/qupath-extension-stardist)).
- **Optionally, the InstanSeg extension** ([qupath-extension-instanseg](https://github.com/qupath/qupath-extension-instanseg)). InstanSeg needs PyTorch through Deep Java Library: open InstanSeg once in the QuPath GUI and let it download PyTorch.

Headless QuPath loads extensions only from its user directory. If you have never opened the GUI, run this once:

```bash
rns setup-qupath --user-dir "C:/Users/<you>/QuPath/v0.7"
```

**Windows note:** QuPath started as a child of Microsoft Store (packaged) Python sees a virtualised registry and loads no extensions. If `detect` reports that StarDist is missing, add `--print-only`. It prints a PowerShell command, with the Java heap set, to run yourself.

## Use

`boxes.csv` lists box centres in full-resolution pixels:

```
x,y,box_um
10234.5,8812.0,120
11020.0,9031.5,120
```

```bash
rns run --image slide.svs --boxes runs/boxes.csv --out runs/slide1 \
        --stardist-model models/he_heavy_augment.pb \
        --instanseg-model models/brightfield_nuclei
```

`run` executes `detect` and then `refine`. You can also run the two steps separately; `rns refine --out runs/slide1` re-runs only the Python step.

Keep `--out` and your box files under `runs/`. The outputs contain crops of the slide, and `.gitignore` covers `runs/` and the output file names.

The output goes to `runs/slide1/refined/`:
- `nuclei.geojson`: import it into QuPath with File > Import objects. Its classes are `Nucleus: sharp`, `Nucleus: intermediate` and `Nucleus: blurred`, and each object carries its measurements, including the `nucleus` number used in the CSV.
- `nuclei.csv`: one row per nucleus, with box, nucleus number, state, centroid (in full-image pixels), area, perimeter, circularity, solidity, convexity, mean nuclear OD, FM and relative focus.
- `overlay_box_####.png`: yellow is sharp, orange is intermediate and cyan is blurred.
- `run_info.json`: classifier accuracy, focus references, counts and all parameters.

You can also call it from Python:

```python
from rns import refine_image, Params
nuclei, info = refine_image(regions, um=0.25, params=Params())
# regions (all from one image): [{"rgb": HxWx3 uint8, "origin": (x0, y0),
#                                 "stardist": [...], "stardist_coarse": [...], "instanseg": [...]}]
```

## Settings of the ASC 2026 poster analyses

The poster ("An Open-Source, Mostly 3D-Printed Slide Scanner for Cytology", ASC 2026) used the defaults above with one change:
only objects that both detectors find start the refinement (coarse StarDist outlines whose centre lies inside an InstanSeg
outline). On Pap tests, objects found by only one detector were mostly folds, cell edges and debris.

```bash
echo '{"seed_mode": "coarse_agree"}' > poster.json
rns refine --out runs/my_slide --params poster.json   # the detect run needs --instanseg-model
```

The nucleus filter still sees all three detectors' outlines. Pap nucleus counts on the poster also left out objects under
20 µm² (smaller than a superficial squamous nucleus); that cutoff was applied in the analysis, not in this package.

## Retraining the nucleus filter

1. Run `rns refine`, then open `refined/nucleus_filter_table.csv`.
2. Add a `label` column with 1 for a nucleus and 0 for anything else. Grade at least about 100 objects across several slides, and look at the objects on the overlays.
3. Train a new model:

   ```bash
   python scripts/train_nucleus_filter.py runs/*/refined/nucleus_filter_table.csv --out models/my_filter.json --group-col box_id
   ```

4. Use the model with `rns refine --params p.json`, where `p.json` contains `{"nucleus_filter_model": "models/my_filter.json"}`. To switch the filter off, use `{"nucleus_filter": false}`.

## Limits

- **The focus flag is not a pure focus measure.** The edge gradient also depends on edge contrast. A sharp nucleus sitting on dense cytoplasm, or deep inside a crowded group, can read as intermediate or blurred, and large high-N:C nuclei are the most affected. Use the flag to describe or stratify; do not use it to delete nuclei. `Params.focus_blob_area_vs_median` can drop large blurred objects, but it is off by default because on Pap it removed real lesional nuclei.
- **Packaged nucleus filter:** it was trained on one project's Pap tests and urine slides (including HSIL and LSIL), scanned on two scanners. With other stains, preparations or scanners, check a random sample of detections and retrain if needed. The strict default keeps only clear nuclei: it removes many real but less clear nuclei (it kept about 60 % of the real nuclei in testing).
- **Over-sampled images:** images with pixels finer than 0.177 µm are refined at 0.177 µm (`Params.work_pixel_um`; 0 = native). Outlines are returned in the original pixels.
- **Classifier and focus reference:** both are per image. If an image has very few nuclei (under about 15), both are rough; `run_info.json` reports this.
- **Uneven chromatin in large nuclei:** the darkness outline uses one threshold per nucleus. A large nucleus with a dark part and a much paler part (e.g. some LSIL nuclei) can be outlined along its dark part only, or with a notch where the chromatin is pale. Area agreement between scanners held when the same nuclei were measured with the detector's own outlines, but contour-shape measures (solidity, convexity, irregularity) of such nuclei are unreliable.
- **Colour rendering:** scanners render the nucleus/cytoplasm boundary with different contrast, so the same nucleus can come out somewhat larger on a scanner that shows less contrast there.
- **Overlapping nuclei:** overlaps are resolved by nearest centroid, so where nuclei overlap, the boundary between them is approximate.
- **Not validated for diagnosis.** It is a research tool.

## Cite the detectors

- **StarDist:** Schmidt U, Weigert M, Broaddus C, Myers G. *Cell detection with star-convex polygons.* MICCAI 2018. Also Weigert M, Schmidt U, Haase R, Sugawara K, Myers G. *Star-convex polyhedra for 3D object detection and segmentation in microscopy.* WACV 2020.
- **InstanSeg:** Goldsborough T, et al. *InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation.* arXiv:2408.15954 (2024).
- **QuPath:** Bankhead P, et al. *QuPath: Open source software for digital pathology image analysis.* Sci Rep 7, 16878 (2017).

The models are not included in this repository. Check the licences of the StarDist model and of the InstanSeg model's training data (listed in its README) before you use them.

## Licence

BSD 3-Clause; see [LICENSE](LICENSE). StarDist, InstanSeg and QuPath are separate projects under their own licences, and
the models they use are downloaded separately.
