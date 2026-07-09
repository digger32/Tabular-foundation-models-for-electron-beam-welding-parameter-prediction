#!/usr/bin/env python3
"""
Fetch the public welding datasets for the external-validity leg. Run ON THE SERVER
(open internet); this repo's sandbox cannot reach dataset hosts.

For each entry in data/datasets.yaml with status "TO FETCH":
  - if a direct `url` is present, download it to data/<path>;
  - otherwise print where to obtain it (DOI / source) and what columns to map.

After fetching, open data/datasets.yaml and confirm `path`, `inputs`, `targets`
for each set against the actual file header, then add the token to --datasets.

Usage:
    python fetch_data.py            # attempt downloads + print instructions
    python fetch_data.py --check    # only report which datasets are ready
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parent / "data"

# Add direct CSV URLs here as they are confirmed (raw file links only). Both public
# sets are hosted behind publisher sites (ScienceDirect/Mendeley), so there is no
# stable hot-link: either download once by hand and drop into data/, or paste a raw
# Mendeley file URL (data.mendeley.com/public-files/...) here to auto-fetch.
DIRECT_URLS = {
    # "gmaw_e1": "https://data.mendeley.com/public-files/.../file_downloaded",
    # "gmaw_e2": "https://data.mendeley.com/public-files/.../file_downloaded",
}


def ready(spec):
    p = DATA / spec.get("path", "")
    return spec.get("path") and p.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    reg = yaml.safe_load((DATA / "datasets.yaml").read_text())

    print("=" * 68)
    for name, spec in reg.items():
        if name == "ebw":
            print(f"[ok]    {name:12s} vendored ({spec['path']})"); continue
        if ready(spec):
            print(f"[ok]    {name:12s} present ({spec['path']})"); continue
        if a.check:
            print(f"[MISS]  {name:12s} not present — {spec.get('source','')}"); continue
        url = DIRECT_URLS.get(name)
        if url:
            dest = DATA / spec["path"]
            try:
                print(f"[get]   {name:12s} downloading -> {dest.name}")
                urllib.request.urlretrieve(url, dest)
                print(f"[ok]    {name:12s} downloaded")
            except Exception as e:
                print(f"[FAIL]  {name:12s} download error: {e}")
        else:
            print(f"[TODO]  {name:12s} no direct URL. Obtain from:\n"
                  f"           {spec.get('source','')}\n"
                  f"        then save as data/{spec['path']} and confirm inputs/targets "
                  f"in datasets.yaml.")
    print("=" * 68)
    reg2 = yaml.safe_load((DATA / "datasets.yaml").read_text())
    have = [n for n, s in reg2.items() if n == "ebw" or ready(s)]
    print(f"ready datasets: {have}")
    if len(have) < 2:
        print("external-validity leg (gate B1) needs >=1 public set — fetch before the final run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
