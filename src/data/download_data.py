"""Download the Cats vs Dogs dataset into data/raw/.

Source: the official Microsoft "Kaggle Cats and Dogs Dataset" (25,000 JPEGs,
12,500 per class) -- https://www.microsoft.com/en-us/download/details.aspx?id=54765

The full archive is ~825 MB, which is far more than a student assignment needs.
Instead of downloading it all, we use HTTP Range requests to read only the zip's
central directory and then only the byte ranges of the images we actually want
(params.yaml -> data.max_images_per_class per class). Because zipfile seeks
between entries, keep the read-ahead buffer small or each small image costs a
whole buffer-sized fetch.

Every image is verified and re-saved as RGB JPEG, which silently drops the
handful of corrupt files known to exist in this dataset (e.g. Cat/666.jpg).

Layout produced:
    data/raw/cat/cat_00000.jpg ...
    data/raw/dog/dog_00000.jpg ...

Run:  python src/data/download_data.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_params, resolve  # noqa: E402

DATASET_URL = (
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/"
    "kagglecatsanddogs_5340.zip"
)
# Folder inside the zip for each class name in params.yaml
ZIP_FOLDER = {"cat": "PetImages/Cat/", "dog": "PetImages/Dog/"}
USER_AGENT = "Mozilla/5.0 (mlops-cats-dogs dataset fetcher)"


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file object backed by HTTP Range requests.

    This lets zipfile.ZipFile treat a remote archive as a local file while
    only transferring the byte ranges it actually asks for.
    """

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.bytes_read = 0
        with urllib.request.urlopen(self._request("bytes=0-0"), timeout=60) as resp:
            if resp.status != 206:
                raise RuntimeError("Server does not support HTTP Range requests")
            self.size = int(resp.headers["Content-Range"].split("/")[-1])

    def _request(self, byte_range: str) -> urllib.request.Request:
        return urllib.request.Request(
            self.url, headers={"User-Agent": USER_AGENT, "Range": byte_range}
        )

    # --- io.RawIOBase interface ---
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _fetch(self, n: int) -> bytes:
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        with urllib.request.urlopen(
            self._request(f"bytes={self.pos}-{end}"), timeout=180
        ) as resp:
            data = resp.read()
        self.pos += len(data)
        self.bytes_read += len(data)
        return data

    def read(self, n: int = -1) -> bytes:
        return self._fetch(self.size - self.pos if n is None or n < 0 else n)

    def readinto(self, buf) -> int:  # required by io.BufferedReader
        data = self._fetch(len(buf))
        buf[: len(data)] = data
        return len(data)


def already_complete(out_dir: Path, wanted: int) -> bool:
    return out_dir.is_dir() and len(list(out_dir.glob("*.jpg"))) >= wanted


def main() -> None:
    params = load_params()
    classes = params["data"]["classes"]
    wanted = int(params["data"]["max_images_per_class"])
    raw_dir = resolve(params["data"]["raw_dir"])

    if all(already_complete(raw_dir / c, wanted) for c in classes):
        print(f"[skip] {raw_dir} already has >= {wanted} images per class.")
        return

    print(f"Opening remote archive (range requests only)\n  {DATASET_URL}")
    remote = HTTPRangeFile(DATASET_URL)
    print(f"  archive size: {remote.size / 1e6:.1f} MB")
    archive = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16))
    print(f"  central directory read ({remote.bytes_read / 1e6:.1f} MB transferred)")

    totals = {}
    for class_name in classes:
        prefix = ZIP_FOLDER[class_name]
        members = sorted(
            (n for n in archive.namelist()
             if n.startswith(prefix) and n.lower().endswith(".jpg")),
            key=lambda n: int(Path(n).stem) if Path(n).stem.isdigit() else 0,
        )
        out_dir = raw_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = skipped = 0
        for member in members:
            if saved >= wanted:
                break
            try:
                blob = archive.read(member)
                image = Image.open(io.BytesIO(blob))
                image.verify()                       # detect truncated/corrupt files
                image = Image.open(io.BytesIO(blob)).convert("RGB")
                image.save(out_dir / f"{class_name}_{saved:05d}.jpg", "JPEG", quality=95)
                saved += 1
            except Exception:
                skipped += 1
                continue
            if saved % 250 == 0:
                print(f"  {class_name}: {saved}/{wanted} "
                      f"({remote.bytes_read / 1e6:.0f} MB downloaded)")

        totals[class_name] = saved
        print(f"[done] {class_name}: saved {saved}, skipped {skipped} corrupt "
              f"-> {out_dir}")

    print(f"\nTotal downloaded: {remote.bytes_read / 1e6:.1f} MB "
          f"(vs {remote.size / 1e6:.0f} MB for the full archive)")
    print("Images per class:", totals)


if __name__ == "__main__":
    main()
