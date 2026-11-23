"""Extract two fixed HSV histogram configurations from the chapter 1 manifests."""
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIGS = ((8, 8, 8), (16, 8, 8))


def hsv_histogram(hsv, bins):
    """Return a C-order H/S/V joint histogram with unit L1 norm."""
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                             [0, 180, 0, 256, 0, 256]).ravel()
    return histogram / histogram.sum()


def extract_color(image, bins=(8, 8, 8)):
    """Convert an OpenCV uint8 BGR image to a normalized color feature."""
    return hsv_histogram(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), bins)


def extract_dataset(dataset):
    manifest = ROOT / "results" / "data_audit" / f"{dataset}_split.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = np.array([r["sample_id"] for r in rows])
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate sample IDs in {manifest}")
    labels = np.array([r["label"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    if set(splits) != {"train", "test"}:
        raise ValueError(f"Unexpected splits in {manifest}")
    features = [np.empty((len(rows), int(np.prod(b))), np.float32) for b in CONFIGS]
    histogram_seconds = np.zeros(2)
    shared_seconds = 0.0
    examples = {}
    started = perf_counter()
    for i, row in enumerate(rows):
        tick = perf_counter()
        path = ROOT / "data" / "raw" / row["path"]
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode sample {row['sample_id']}: {path}")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        shared_seconds += perf_counter() - tick
        # Alternate order to reduce systematic first-configuration timing effects.
        for j in ((0, 1) if i % 2 == 0 else (1, 0)):
            tick = perf_counter()
            features[j][i] = hsv_histogram(hsv, CONFIGS[j])
            histogram_seconds[j] += perf_counter() - tick
        if row["split"] == "train" and row["label"] not in examples:
            examples[row["label"]] = (i, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if (i + 1) % 2000 == 0:
            print(f"{dataset}: {i + 1}/{len(rows)}", flush=True)
    loop_seconds = perf_counter() - started
    cache = ROOT / "results" / "cache" / "color"
    output = ROOT / "results" / "color"
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for j, bins in enumerate(CONFIGS):
        x = features[j]
        if not np.isfinite(x).all() or (x < 0).any() or not np.allclose(x.sum(axis=1), 1, atol=1e-6):
            raise ValueError(f"Invalid normalized features for {dataset} {bins}")
        name = f"{dataset}_{'x'.join(map(str, bins))}"
        # Publish each completed cache with a single rename.
        temporary = cache / f"{name}.tmp.npz"
        np.savez_compressed(temporary, X=x, sample_id=ids, label=labels, split=splits,
                            bins=np.array(bins), manifest_sha256=np.array(hashlib.sha256(manifest.read_bytes()).hexdigest()))
        temporary.replace(cache / f"{name}.npz")
        summary.append(dict(dataset=dataset, bins="x".join(map(str, bins)),
                            samples=len(rows), train=int((splits == "train").sum()),
                            test=int((splits == "test").sum()), dimension=x.shape[1],
                            decode_hsv_seconds=shared_seconds,
                            histogram_seconds=float(histogram_seconds[j]),
                            combined_seconds=float(shared_seconds + histogram_seconds[j]),
                            loop_both_seconds=loop_seconds,
                            zero_fraction=float(np.count_nonzero(x == 0) / x.size),
                            feature_mib=x.nbytes / 1024**2, failures=0))
    fig, axes = plt.subplots(len(examples), 3, figsize=(12, len(examples)*2.2), squeeze=False)
    example_rows = []
    for axis, (label, (i, image)) in zip(axes, examples.items()):
        axis[0].imshow(image)
        axis[0].set_title(f"{label} | train")
        axis[0].axis("off")
        for j in range(2):
            axis[j+1].plot(features[j][i], linewidth=.7)
            axis[j+1].set_title(f"{CONFIGS[j]} | {features[j].shape[1]} dimensions")
            axis[j+1].set_xlabel("Flattened H/S/V bin index")
            axis[j+1].set_ylabel("Pixel fraction")
        example_rows.append(dict(label=label, sample_id=str(ids[i]), split="train"))
    fig.tight_layout()
    fig.savefig(output / f"{dataset}_examples.png", dpi=130)
    plt.close(fig)
    (output / f"{dataset}_examples.json").write_text(json.dumps(example_rows, indent=2), encoding="utf-8")
    return summary


def main():
    cv2.setNumThreads(1)
    summaries = []
    for dataset in ("food101", "imagenette"):
        summaries.extend(extract_dataset(dataset))
    output = ROOT / "results" / "color"
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    metadata = dict(python=platform.python_version(), numpy=np.__version__,
                    opencv=cv2.__version__, matplotlib=matplotlib.__version__,
                    platform=platform.platform(), processor=platform.processor(),
                    opencv_threads=1, seed=42, randomness="none",
                    resize="none", normalization="L1", order="H,S,V; C-order flatten",
                    timing="One pass; read/decode/HSV shared; histogram time separate; excludes cache writing and plots",
                    failure_policy="Raise with sample ID; never skip or insert zero features",
                    configuration_policy="Two fixed candidates; choose using training-only validation in chapter 3")
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
