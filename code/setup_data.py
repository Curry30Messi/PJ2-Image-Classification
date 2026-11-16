"""Verify and unpack the two course archives. Requires Python 3.12 or later."""
import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

DATA = Path(__file__).resolve().parent / "data"
ARCHIVE_ROOTS = {
    "PJ2--food_101_dataset.tgz": "food-101-coursework-6classes",
    "PJ2--imagenette2-320.tgz": "imagenette2-320",
}


def setup(archive_dir, destination):
    expected = {}
    for line in (DATA / "archives.sha256").read_text().splitlines():
        digest, name = line.split()
        expected[name] = digest

    for name in ARCHIVE_ROOTS:
        archive = archive_dir / name
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected[name]:
            raise ValueError(f"SHA-256 mismatch for {archive}. Use the course archive.")

    destination.mkdir(parents=True, exist_ok=True)
    for name, top in ARCHIVE_ROOTS.items():
        target = destination / top
        if target.exists():
            raise FileExistsError(f"Target already exists: {target}")
    with TemporaryDirectory(dir=destination) as scratch:
        staging = Path(scratch)
        for name, top in ARCHIVE_ROOTS.items():
            with tarfile.open(archive_dir / name, "r:gz") as archive:
                archive.extractall(staging, filter="data")
            if not (staging / top).is_dir():
                raise ValueError(f"Missing expected directory {top} in {name}")
        for top in ARCHIVE_ROOTS.values():
            shutil.move(str(staging / top), destination / top)
            print(destination / top)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path, help="Directory containing both course .tgz files")
    parser.add_argument("--destination", type=Path, default=DATA / "raw")
    args = parser.parse_args()
    try:
        setup(args.archive_dir, args.destination)
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
