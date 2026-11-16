"""Build and audit the fixed Food-101 and Imagenette course splits."""
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

BASE = Path(__file__).resolve().parent
FOOD_CLASSES = ("sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles")
IMAGENETTE_CLASSES = {
    "n01440764": "tench", "n02102040": "English springer",
    "n02979186": "cassette player", "n03000684": "chain saw",
    "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump",
    "n03445777": "golf ball", "n03888257": "parachute",
}
IMAGENETTE_COUNTS = {
    "train": (963, 955, 993, 858, 941, 956, 961, 931, 951, 960),
    "test": (387, 395, 357, 386, 409, 394, 389, 419, 399, 390),
}
SPLIT_FIELDS = ["sample_id", "split", "path", "label"]


def food_rows(raw):
    root = raw / "food-101-coursework-6classes"
    for split in ("train", "test"):
        entries = (root / "meta" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        for entry in entries:
            label, slash, image_id = entry.strip().partition("/")
            if label not in FOOD_CLASSES or not slash or not image_id.isdecimal():
                raise ValueError(f"Unexpected Food-101 metadata entry: {entry!r}")
            relpath = (root / "images" / label / f"{image_id}.jpg").relative_to(raw)
            yield dict(sample_id=f"{label}/{image_id}", split=split,
                       path=relpath.as_posix(), label=label)


def imagenette_rows(raw):
    root = raw / "imagenette2-320"
    for split, directory in (("train", "train"), ("test", "val")):
        split_root = root / directory
        folders = {path.name for path in split_root.iterdir() if path.is_dir()}
        if folders != IMAGENETTE_CLASSES.keys():
            raise ValueError(f"Unexpected Imagenette class folders in {split_root}")
        for synset, label in IMAGENETTE_CLASSES.items():
            for path in sorted((split_root / synset).iterdir()):
                if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    yield dict(sample_id=f"{synset}/{path.name}", split=split,
                               path=path.relative_to(raw).as_posix(), label=label)


def validate_split(rows, expected):
    counts = Counter((row["split"], row["label"]) for row in rows)
    if counts != expected:
        raise ValueError(f"Unexpected class counts: {counts - expected}; missing: {expected - counts}")
    ids = Counter(row["sample_id"] for row in rows)
    repeated = [sample_id for sample_id, count in ids.items() if count > 1]
    if repeated:
        raise ValueError(f"Sample IDs reused within or across splits: {repeated[:8]}")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_images(raw, rows):
    stats = []
    digests = {"file_sha256": defaultdict(list), "pixel_sha256": defaultdict(list)}
    for row in rows:
        path = raw / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or not image.size:
            raise ValueError(f"Unable to decode {path}")
        height, width = image.shape[:2]
        file_digest = hashlib.sha256(payload).hexdigest()
        # Include the decoded shape so equal byte streams with different axes do not collide.
        pixel_hash = hashlib.sha256(f"{height}x{width}x3:".encode("ascii"))
        pixel_hash.update(memoryview(image))
        pixels_digest = pixel_hash.hexdigest()
        stats.append({**row, "width": width, "height": height, "short_side": min(width, height),
                      "file_sha256": file_digest, "pixel_sha256": pixels_digest})
        digests["file_sha256"][file_digest].append(row)
        digests["pixel_sha256"][pixels_digest].append(row)

    overlap = []
    for method, bucket in digests.items():
        for digest, items in sorted(bucket.items()):
            if {row["split"] for row in items} == {"train", "test"}:
                overlap.extend({"method": method, "digest": digest, **row} for row in items)
    return stats, overlap


def summarize(dataset, rows, labels, overlaps):
    counts = Counter((row["split"], row["label"]) for row in rows)
    by_class = [
        dict(dataset=dataset, label=label, train=counts["train", label], test=counts["test", label])
        for label in labels
    ]
    summary = []
    for split in ("train", "test"):
        values = [row["short_side"] for row in rows if row["split"] == split]
        p25, median, p75 = np.percentile(values, [25, 50, 75])
        summary.append(dict(dataset=dataset, split=split, images=len(values),
                            short_min_px=min(values), short_p25_px=p25,
                            short_median_px=median, short_p75_px=p75, short_max_px=max(values)))
    check = {
        "decoded_images": len(rows),
        "missing_or_undecodable": 0,
        "shared_sample_ids": 0,
        "cross_split_file_sha256_groups": len({r["digest"] for r in overlaps if r["method"] == "file_sha256"}),
        "cross_split_pixel_sha256_groups": len({r["digest"] for r in overlaps if r["method"] == "pixel_sha256"}),
    }
    return by_class, summary, check


def plot_distributions(output, counts, all_stats):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
    for ax, (dataset, entries) in zip(axes, counts.items(), strict=True):
        positions = np.arange(len(entries))
        for split, offset, color in [("train", -0.18, "#253546"), ("test", 0.18, "#a7b9bf")]:
            ax.barh(positions + offset, [row[split] for row in entries], height=0.34,
                    color=color, label=split.capitalize())
        ax.set(yticks=positions, yticklabels=[row["label"].replace("_", " ") for row in entries],
               xlabel="Images", title=dataset)
        ax.invert_yaxis()
        ax.set_xlim(0, 1090)
        ax.legend(loc="upper right")
    fig.savefig(output / "class_counts.png", dpi=190)
    plt.close(fig)

    labels = []
    groups = []
    for dataset, stats in all_stats.items():
        for split in ("train", "test"):
            labels.append(f"{dataset}\n{split}")
            groups.append([r["short_side"] for r in stats if r["split"] == split])
    fig, ax = plt.subplots(figsize=(7.6, 3.1), layout="constrained")
    ax.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#c6d2d9"}, medianprops={"color": "#222222"},
               whis=(0, 100))
    ax.set_ylabel("Decoded short side (px)")
    ax.grid(axis="y", alpha=0.24)
    fig.savefig(output / "image_sizes.png", dpi=190)
    plt.close(fig)


def run(raw, output):
    specifications = [
        ("food101", "Food-101", food_rows, dict(zip(FOOD_CLASSES, FOOD_CLASSES, strict=True)),
         {(split, label): count for split, count in [("train", 750), ("test", 250)] for label in FOOD_CLASSES}),
        ("imagenette", "Imagenette", imagenette_rows, IMAGENETTE_CLASSES,
         {(split, label): count for split, counts in IMAGENETTE_COUNTS.items()
          for label, count in zip(IMAGENETTE_CLASSES.values(), counts, strict=True)}),
    ]
    audit = {"python": platform.python_version(), "opencv": cv2.__version__,
             "numpy": np.__version__, "matplotlib": matplotlib.__version__, "datasets": {}}
    counts = {}
    image_stats = {}
    sizes = []
    all_overlaps = []
    prepared = {}
    for key, display, load_rows, labels, expected in specifications:
        rows = list(load_rows(raw))
        validate_split(rows, expected)
        stats, overlaps = audit_images(raw, rows)
        by_class, summary, checks = summarize(display, stats, labels.values(), overlaps)
        counts[display] = by_class
        image_stats[display] = stats
        sizes.extend(summary)
        all_overlaps.extend({"dataset": key, **row} for row in overlaps)
        audit["datasets"][key] = checks
        prepared[key] = rows

    output.mkdir(parents=True, exist_ok=True)
    for key, _, _, labels, _ in specifications:
        write_csv(output / f"{key}_split.csv", SPLIT_FIELDS, prepared[key])
        mapping = [{"label_id": i, "source_label": source, "label": label}
                   for i, (source, label) in enumerate(labels.items())]
        write_csv(output / f"{key}_labels.csv", ["label_id", "source_label", "label"], mapping)
    write_csv(output / "class_counts.csv", ["dataset", "label", "train", "test"],
              [row for entries in counts.values() for row in entries])
    write_csv(output / "size_summary.csv", list(sizes[0]), sizes)
    write_csv(output / "exact_cross_split_matches.csv",
              ["dataset", "method", "digest", *SPLIT_FIELDS], all_overlaps)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot_distributions(output, counts, image_stats)
    print(json.dumps(audit, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=BASE / "data" / "raw")
    parser.add_argument("--output", type=Path, default=BASE / "results" / "data_audit")
    args = parser.parse_args()
    run(args.raw, args.output)


if __name__ == "__main__":
    main()
