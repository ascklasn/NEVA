import argparse
import json
import logging
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import skimage.color as sk_color
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from trident import load_wsi
except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard for missing optional deps
    if exc.name == "openslide":
        raise SystemExit(
            "OpenSlide (openslide-python) is required to read the WSI. "
            "Please install the OpenSlide library and the Python bindings before running this script."
        ) from exc
    raise

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is missing

    def tqdm(iterable, **kwargs):
        return iterable


SCORING_RESIZE = 112
SATURATION_THRESHOLD = 0.2
LOW_VALUE_THRESHOLD = 0.6
HIGH_VALUE_THRESHOLD = 0.92
TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10
HSV_PURPLE = 270
HSV_PINK = 330
BLACK_V = 0.12
WHITE_V = 0.95
WHITE_S = 0.10
MAX_BLACK_FRAC = 0.30
MAX_WHITE_FRAC = 0.60
WSI_FILE_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn")
BATCH_SIZE = 64
MAX_TOP_PATCHES = 200
TOP_SELECTION_FRACTION = 0.20
ARTIFACT_SUBDIR = "artifacts"


def slide_already_processed(slide_name, output_dir):
    """Return True if the slide output directory already contains the expected artifacts."""
    if not output_dir.exists():
        return False

    artifact_dir = output_dir / ARTIFACT_SUBDIR
    plot_path = artifact_dir / f"{slide_name}_final_scores.png"
    meta_path = artifact_dir / "selection_info.json"

    if not (artifact_dir.exists() and plot_path.exists() and meta_path.exists()):
        # Fallback to legacy layout where artifacts lived alongside patches.
        plot_path = output_dir / f"{slide_name}_final_scores.png"
        meta_path = output_dir / "selection_info.json"
        if not (plot_path.exists() and meta_path.exists()):
            return False

    try:
        with meta_path.open("r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
    except (OSError, json.JSONDecodeError):
        return False

    expected_patches = int(metadata.get("selected_patch_count", 0))
    if expected_patches <= 0:
        return False

    patch_files = [
        path
        for path in output_dir.glob("*.png")
        if path.is_file() and "_score_" in path.name and not path.name.endswith("_final_scores.png")
    ]
    return len(patch_files) >= expected_patches


def save_final_score_plot(coords, scores, selected_indices, artifact_dir, slide_name):
    """Generate a scatter plot of final scores with selected patches highlighted."""
    coords = np.asarray(coords)
    scores = np.asarray(scores)
    selected_indices = np.asarray(selected_indices, dtype=int)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="viridis", s=10, alpha=0.6)

    if selected_indices.size > 0:
        highlight_coords = coords[selected_indices]
        ax.scatter(
            highlight_coords[:, 0],
            highlight_coords[:, 1],
            facecolors="none",
            edgecolors="red",
            s=40,
            linewidths=0.7,
            label=f"Selected patches ({selected_indices.size})",
        )
        ax.legend(loc="best")

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Final score distribution - {slide_name}")
    fig.colorbar(scatter, ax=ax, label="Final score")
    plot_path = artifact_dir / f"{slide_name}_final_scores.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def normalize_patch_image(patch, target_size):
    """Ensure patches are RGB, resized, and composited on a white background if needed."""
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    if patch.size != target_size:
        patch = patch.resize(target_size, Image.BILINEAR)

    if "A" in patch.getbands():
        patch = patch.convert("RGBA")
        bg = Image.new("RGBA", target_size, (255, 255, 255, 255))
        patch = Image.alpha_composite(bg, patch)
        patch = patch.convert("RGB")
    elif patch.mode != "RGB":
        patch = patch.convert("RGB")

    return patch


def bg_stats_from_hsv(hsv_batch):
    s = hsv_batch[..., 1]
    v = hsv_batch[..., 2]
    black_mask = v < BLACK_V
    white_mask = (v > WHITE_V) & (s < WHITE_S)
    black_frac = black_mask.mean(axis=(1, 2))
    white_frac = white_mask.mean(axis=(1, 2))
    return black_frac, white_frac


def reject_bad_background(hsv_batch):
    black_frac, white_frac = bg_stats_from_hsv(hsv_batch)
    return (black_frac > MAX_BLACK_FRAC) | (white_frac > MAX_WHITE_FRAC)


def estimate_tissue_ratio_batch(rgb_batch):
    if isinstance(rgb_batch, list):
        batch_array = np.array(rgb_batch)
    else:
        batch_array = rgb_batch

    batch_array_float = batch_array.astype(np.float32) / 255.0
    hsv_batch = sk_color.rgb2hsv(batch_array_float)
    saturation = hsv_batch[..., 1]
    value = hsv_batch[..., 2]

    tissue_mask = ((saturation > SATURATION_THRESHOLD) & (value < HIGH_VALUE_THRESHOLD)) | (value < LOW_VALUE_THRESHOLD)
    tissue_ratios = np.mean(tissue_mask, axis=(1, 2))
    return tissue_ratios.tolist()


def tissue_quantity_factor_batch(tissue_percents):
    tissue_percents = np.array(tissue_percents)
    factors = np.zeros_like(tissue_percents)

    factors[tissue_percents >= TISSUE_HIGH_THRESH] = 1.0
    mask_medium = (tissue_percents >= TISSUE_LOW_THRESH) & (tissue_percents < TISSUE_HIGH_THRESH)
    factors[mask_medium] = 0.2
    mask_low = (tissue_percents > 0) & (tissue_percents < TISSUE_LOW_THRESH)
    factors[mask_low] = 0.1

    return factors


def hsv_purple_pink_factor_batch(rgb_batch, hsv_batch=None):
    if hsv_batch is None:
        rgb_batch_float = rgb_batch.astype(np.float32) / 255.0
        hsv_batch = sk_color.rgb2hsv(rgb_batch_float)

    hues_batch = (hsv_batch[..., 0] * 360).astype(np.int32)
    factors = np.zeros(len(rgb_batch))

    for i in range(len(rgb_batch)):
        hues = hues_batch[i].flatten()
        hues_filtered = hues[(hues >= 260) & (hues <= 340)]

        if len(hues_filtered) == 0:
            factors[i] = 0
            continue

        pu_dev = np.sqrt(np.mean(np.abs(hues_filtered - HSV_PURPLE) ** 2))
        pi_dev = np.sqrt(np.mean(np.abs(hues_filtered - HSV_PINK) ** 2))
        avg_factor = (340 - np.mean(hues_filtered)) ** 2

        factors[i] = 0 if pu_dev == 0 else (pi_dev / pu_dev) * avg_factor

    return factors


def hsv_saturation_and_value_factor_batch(rgb_batch, hsv_batch=None):
    if hsv_batch is None:
        rgb_batch_float = rgb_batch.astype(np.float32) / 255.0
        hsv_batch = sk_color.rgb2hsv(rgb_batch_float)

    s_batch = hsv_batch[..., 1]
    v_batch = hsv_batch[..., 2]
    s_std = np.std(s_batch, axis=(1, 2))
    v_std = np.std(v_batch, axis=(1, 2))

    factors = np.ones(len(rgb_batch))
    mask_both = (s_std < 0.05) & (v_std < 0.05)
    mask_s_only = (s_std < 0.05) & ~mask_both
    mask_v_only = (v_std < 0.05) & ~mask_both

    factors[mask_both] = 0.4
    factors[mask_s_only] = 0.7
    factors[mask_v_only] = 0.7

    return factors**2


def score_tile_batch(np_tiles, tissue_percents, hsv_batch=None):
    if hsv_batch is None:
        np_tiles_float = np_tiles.astype(np.float32) / 255.0
        hsv_batch = sk_color.rgb2hsv(np_tiles_float)

    reject_mask = reject_bad_background(hsv_batch)
    color_factors = hsv_purple_pink_factor_batch(np_tiles, hsv_batch)
    s_and_v_factors = hsv_saturation_and_value_factor_batch(np_tiles, hsv_batch)
    quantity_factors = tissue_quantity_factor_batch(tissue_percents)

    combined_factors = color_factors * s_and_v_factors * quantity_factors
    tissue_percents_array = np.array(tissue_percents)
    scores = (tissue_percents_array**2) * np.log1p(combined_factors) / 1000.0
    scaled_scores = 1.0 - (1.0 / (1.0 + scores))
    scaled_scores[reject_mask] = 0.0
    return scaled_scores * 100


def find_wsi_path(slide_id, slide_roots, extensions=WSI_FILE_EXTENSIONS):
    """Return the first matching WSI path for ``slide_id`` under the provided roots."""
    matches = []
    for root in slide_roots:
        root_path = Path(root)
        if not root_path.exists():
            logging.warning("WSI root %s does not exist; skipping.", root_path)
            continue

        for ext in extensions:
            matches.extend(root_path.rglob(f"{slide_id}{ext}"))

    if not matches:
        searched_roots = ", ".join(str(Path(r)) for r in slide_roots)
        raise FileNotFoundError(f"WSI for slide_id '{slide_id}' not found under [{searched_roots}].")

    if len(matches) > 1:
        logging.warning("Multiple WSIs found for %s; using %s", slide_id, matches[0])

    return matches[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and score top patches from slide features.")
    parser.add_argument(
        "--feat-root",
        type=Path,
        default=Path(
            "trident_processed/10x_384px_0px_overlap/slide_features_feather"
        ),
        help="Directory with slide feature .h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/scratch/xiangjx/data/top200_patches"),
        help="Directory to save top ranking patches.",
    )
    parser.add_argument(
        "--wsi-root",
        dest="wsi_roots",
        action="append",
        default=[
            "wsi/ertong",
            "wsi/qianzhan_24_06/20240607神母",
        ],
        help="Root directory containing WSIs; can be provided multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of slides to process (useful for debugging).",
    )
    parser.add_argument(
        "--partition-count",
        type=int,
        default=1,
        help="Total number of partitions to split slides into (default: 1).",
    )
    parser.add_argument(
        "--partition-index",
        type=int,
        default=0,
        help="Zero-based partition index to process (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    feat_root = args.feat_root.expanduser().resolve()

    if not feat_root.exists():
        raise SystemExit(f"Feature root {feat_root} does not exist.")

    slide_ids = [path.stem for path in sorted(feat_root.glob("*.h5"))]
    if args.limit is not None:
        slide_ids = slide_ids[: args.limit]

    print(f"Found {len(slide_ids)} slides to process in {feat_root}")

    partition_count = int(args.partition_count)
    partition_index = int(args.partition_index)

    if partition_count <= 0:
        raise SystemExit("--partition-count must be greater than 0.")
    if partition_index < 0 or partition_index >= partition_count:
        raise SystemExit("--partition-index must be in the range [0, partition-count).")

    slide_ids = [slide_id for idx, slide_id in enumerate(slide_ids) if idx % partition_count == partition_index]
    print(f"Processing partition {partition_index + 1}/{partition_count} with {len(slide_ids)} slides.")

    for slide_id in tqdm(slide_ids):
        h5_path = feat_root / f"{slide_id}.h5"
        try:
            wsi_path = find_wsi_path(slide_id, args.wsi_roots).resolve()
        except FileNotFoundError as exc:
            logging.error("%s - skipping slide.", exc)
            continue

        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            attns = np.abs(f["attentions"][:])
            attrs = dict(f["coords"].attrs)

        slide_name = attrs.get("name", wsi_path.stem)
        output_dir = args.output_dir / slide_name

        if slide_already_processed(slide_name, output_dir):
            logging.info("Skipping %s - already processed.", slide_name)
            continue

        wsi = load_wsi(slide_path=str(wsi_path), lazy_init=False)
        patch_size = int(attrs.get("patch_size", 256))
        patch_size_level0 = int(attrs.get("patch_size_level0", patch_size))

        target_downsample = patch_size_level0 / patch_size
        level_downsamples = np.asarray(wsi.level_downsamples, dtype=float)
        level_idx = int(np.argmin(np.abs(level_downsamples - target_downsample)))
        level_downsample = float(level_downsamples[level_idx])
        patch_size_at_level = max(1, int(round(patch_size_level0 / level_downsample)))

        print(
            f"Selected level {level_idx} with downsample {level_downsample:.4f} "
            f"(target {target_downsample:.4f}); patch size at level {patch_size_at_level}px."
        )

        coords_count = len(coords)
        tile_scores = np.zeros(coords_count, dtype=float)

        for start in tqdm(range(0, coords_count, BATCH_SIZE), desc="Scoring tiles", unit="batch"):
            end = min(start + BATCH_SIZE, coords_count)
            coords_batch = coords[start:end]

            scoring_tiles = []
            for x, y in coords_batch:
                location = (int(x), int(y))
                patch = wsi.read_region(location, level_idx, (patch_size_at_level, patch_size_at_level), read_as="pil")
                patch = normalize_patch_image(patch, (patch_size, patch_size))

                scoring_patch = patch
                if patch.size != (SCORING_RESIZE, SCORING_RESIZE):
                    scoring_patch = patch.resize((SCORING_RESIZE, SCORING_RESIZE), Image.BILINEAR)

                scoring_tiles.append(np.asarray(scoring_patch, dtype=np.uint8))

            if not scoring_tiles:
                continue

            scoring_array = np.stack(scoring_tiles, axis=0)
            tissue_percents = estimate_tissue_ratio_batch(scoring_array)
            batch_scores = score_tile_batch(scoring_array, tissue_percents)
            tile_scores[start:end] = batch_scores

        attn_scores = np.asarray(attns, dtype=float).reshape(-1)
        final_scores = attn_scores * tile_scores

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving patches to {output_dir}")

        sorted_indices = np.argsort(final_scores)[::-1]
        top_twenty_count = max(1, int(np.ceil(len(sorted_indices) * TOP_SELECTION_FRACTION)))
        top_twenty_indices = sorted_indices[:top_twenty_count]

        if len(top_twenty_indices) >= MAX_TOP_PATCHES:
            rng = np.random.default_rng()
            selected_indices = rng.choice(top_twenty_indices, size=MAX_TOP_PATCHES, replace=False)
        else:
            selected_indices = sorted_indices[: min(MAX_TOP_PATCHES, len(sorted_indices))]

        selected_indices = np.asarray(selected_indices, dtype=int)
        if selected_indices.size > 0:
            order = np.argsort(final_scores[selected_indices])[::-1]
            selected_indices = selected_indices[order]

        artifact_dir = output_dir / ARTIFACT_SUBDIR
        artifact_dir.mkdir(parents=True, exist_ok=True)

        save_final_score_plot(coords, final_scores, selected_indices, artifact_dir, slide_name)

        selection_metadata = {
            "slide_id": slide_id,
            "slide_name": slide_name,
            "selected_patch_count": int(selected_indices.size),
            "total_patches": int(len(sorted_indices)),
            "top_fraction": TOP_SELECTION_FRACTION,
            "selection_mode": (
                "random_within_top_fraction" if len(top_twenty_indices) >= MAX_TOP_PATCHES else "top_overall"
            ),
        }
        metadata_path = artifact_dir / "selection_info.json"
        with metadata_path.open("w", encoding="utf-8") as meta_file:
            json.dump(selection_metadata, meta_file, indent=2)

        for rank, idx in enumerate(
            tqdm(selected_indices, desc="Extracting patches", unit="patch", total=selected_indices.size)
        ):
            x, y = coords[idx]
            location = (int(x), int(y))
            patch = wsi.read_region(location, level_idx, (patch_size_at_level, patch_size_at_level), read_as="pil")
            patch = normalize_patch_image(patch, (patch_size, patch_size))
            attention_score = float(attn_scores[idx])
            tile_score = float(tile_scores[idx])
            final_score = float(final_scores[idx])
            patch.save(
                output_dir
                / f"{slide_name}_{rank:05d}_attn_{attention_score:.3f}_tile_{tile_score:.3f}_score_{final_score:.3f}.png"
            )

        if hasattr(wsi, "img") and hasattr(wsi.img, "close"):
            wsi.img.close()
