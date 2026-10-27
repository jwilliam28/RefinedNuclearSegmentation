"""All tunable settings in one place. Distances are in micrometres so the same settings work at any pixel size."""
from dataclasses import dataclass, asdict


@dataclass
class Params:
    # union of the two detectors
    instanseg_add_max_cover: float = 0.10   # an InstanSeg nucleus is added only where StarDist covers < 10 % of it

    # coarse consolidation (a nucleus split into fragments by the fine StarDist pass)
    coarse_min_solidity: float = 0.90
    coarse_min_inside: float = 0.60         # each fine piece must lie >= 60 % inside the coarse instance
    coarse_dominant_frac: float = 0.55      # and the largest piece must be >= 55 % of the coarse area
    coarse_no_gap: float = 0.85             # or: no pale gap between the pieces (OD on the cuts >= 0.85 x paler piece; 0 = off)
    coarse_no_gap_max_elong: float = 1.4    # ... and only if the coarse outline's major/minor axis ratio is <= 1.4

    # colour classifier: nucleus vs background (cytoplasm ring + glass pooled), trained per image on the detections
    cyto_ring_um: tuple = (1.5, 4.0)
    glass_luma_frac: float = 0.92           # glass = luma > 98th percentile x 0.92
    samples_per_class: int = 4000

    # images finer than this (by > 10 %) are refined at this pixel size: over-sampled scans gain no nuclear detail, and the
    # focus measure works at 0.5 um (0 = always the native pixel size)
    work_pixel_um: float = 0.177

    # outside-in radial boundary
    rays: int = 128
    ray_step_um: float = 0.25
    ray_margin_um: float = 1.5              # rays start at least this far outside the detector outline
    grow_um: float = 0.0                    # optional growth along the nucleus probability up to this far (0 = off; off by
                                            # default since 2026-10-26: on Pap it changed outlines little and sometimes added
                                            # cytoplasm - cytopathologist review of random cases)
    grow_min_closed: float = 0.70           # growth kept only if >= 70 % of its new border is cytoplasm/glass
    grow_min_od_frac: float = 0.60          # and only through pixels >= 0.6 x the core's nuclear OD
    ray_sustain_um: float = 0.7             # first sustained run of p > 0.5 coming inward
    ray_snap_um: float = 1.0                # then snap to the steepest rise of nuclear OD within +/- 1 um
    darkness_preference: float = 0.0        # the darkness (Otsu) outline is used when its edge is >= this x the radial
                                            # outline's; 0 = darkness outline first (radial only where it fails). Default 0
                                            # since 2026-10-26: on Pap the darkness outline was judged better and the radial
                                            # outline picked up cytoplasm halos, mostly on the blurrier commercial plane
    darkness_ring_frac: float = 0.5         # the darkness threshold is estimated on the outline region plus a surround ring
                                            # of max(1 um, this x the detector outline's equivalent radius), so it falls
                                            # between nucleus and cytoplasm even for large nuclei (2026-10-26)
    darkness_glass_od: float = 0.08         # glass (nuclear OD <= this) is left out of that sample

    # clean-up
    min_area_um2: float = 8.0
    min_mean_prob: float = 0.4
    min_solidity: float = 0.75
    min_contrast: float = 1.15              # nucleus OD >= 1.15 x its own 1-2.5 um surround (glass and neighbours excluded)

    # merging of pieces of one nucleus
    merge_pieces: bool = False              # join touching pieces as 'halves' / 'fragment' (below). Off by default since
                                            # 2026-10-26: on Pap it mostly joined pieces of blurred clumps and cytoplasm
                                            # (cytopathologist review of the merges it made)
    halves_min_solidity: float = 0.93
    halves_max_elongation: float = 1.7
    halves_max_area_vs_median: float = 1.6
    fragment_dominant_frac: float = 0.70
    fragment_min_solidity: float = 0.90
    fragment_no_gap: float = 0.85           # OD along the cut >= 0.85 x the paler piece's mean OD

    # object-level nucleus filter (nucleus_filter.py): removes folds, strands, specks, cytoplasm edges and Candida
    nucleus_filter: bool = True
    nucleus_filter_model: str = ''          # '' = the packaged model (models/nucleus_filter_pap_v1.json); or a path to your own
    nucleus_filter_threshold: float = 0.0   # 0 = the model's own threshold (strict: chosen for ~94 % precision)

    # outline smoothing
    smooth_wavelength_um: float = 2.0       # Fourier harmonics with this wavelength are halved

    # focus: self-normalised edge-gradient measure (a flag, not a filter; see README Limits)
    focus_smooth_um: float = 0.5            # luminance pre-smoothing before the gradient
    focus_band_um: float = 1.0              # +/- band around the outline
    focus_ref_percentile: float = 75.0      # field reference = 75th percentile of the field's nuclei
    focus_min_field_nuclei: int = 5         # fewer nuclei in a field: use the image-level reference
    focus_floor: float = 0.60               # reference floored at 0.60 x the image's median field reference
    focus_sharp_rel: float = 0.75           # FM / reference >= 0.75 = sharp
    focus_blur_rel: float = 0.50            # < 0.50 = blurred; in between = intermediate
    focus_blob_area_vs_median: float = 0.0  # > 0 (e.g. 2.5): blurred nuclei larger than this x the median sharp area
                                            # are classed 'blob' and left out of the outputs. Off by default: on Pap,
                                            # large high-N:C nuclei on dense cytoplasm can read as blurred (see README)
    focus_min_ref_nuclei: int = 15          # fewer nuclei in the whole image: run_info flags the focus as rough

    def to_dict(self):
        return asdict(self)
