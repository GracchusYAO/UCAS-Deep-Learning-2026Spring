from __future__ import annotations
import argparse
import gzip
import shutil
import urllib.error
import urllib.request
from pathlib import Path


MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://raw.githubusercontent.com/mkolod/MNIST/master/",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Download raw MNIST files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root / "Data" / "MNIST" / "raw",
        help="Directory used to store the raw MNIST files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for a single download attempt.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of attempts per mirror.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path, timeout: float) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def extract_gzip_file(gzip_path: Path) -> Path:
    extracted_path = gzip_path.with_suffix("")
    if extracted_path.exists() and extracted_path.stat().st_size > 0:
        print(f"Already extracted, skipping: {extracted_path.name}")
        return extracted_path

    with gzip.open(gzip_path, "rb") as compressed_file, extracted_path.open("wb") as extracted_file:
        shutil.copyfileobj(compressed_file, extracted_file)

    print(f"Extracted to {extracted_path}")
    return extracted_path


def ensure_mnist(root: Path, timeout: float, retries: int) -> None:
    root.mkdir(parents=True, exist_ok=True)

    for file_name in MNIST_FILES:
        destination = root / file_name
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Already exists, skipping: {destination.name}")
            continue

        last_error: Exception | None = None
        for mirror in MNIST_MIRRORS:
            url = mirror + file_name
            for attempt in range(1, retries + 1):
                print(f"Downloading {file_name} from {url} (attempt {attempt}/{retries})")
                try:
                    download_file(url, destination, timeout=timeout)
                    print(f"Saved to {destination}")
                    extract_gzip_file(destination)
                    last_error = None
                    break
                except (urllib.error.URLError, OSError) as exc:
                    last_error = exc
                    if destination.exists():
                        destination.unlink()
                    extracted_path = destination.with_suffix("")
                    if extracted_path.exists():
                        extracted_path.unlink()
                    print(f"Failed from {url}: {exc}")
            if last_error is None:
                break

        if last_error is not None:
            raise RuntimeError(f"Could not download {file_name}") from last_error


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    print(f"Downloading MNIST raw files into: {root}")
    ensure_mnist(root, timeout=args.timeout, retries=args.retries)
    print("Download complete.")
    print("Files:")
    for file_name in MNIST_FILES:
        print(f"  - {root / file_name}")
        print(f"  - {(root / file_name).with_suffix('')}")


if __name__ == "__main__":
    main()
