from __future__ import annotations

import argparse
import io
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

GEOM_GCN = {
    "texas": "texas",
    "wisconsin": "wisconsin",
    "cornell": "cornell",
    "actor": "film",
    "chameleon": "chameleon",
    "squirrel": "squirrel",
}

HETEROPHILY_NPZ = {
    "roman_empire": "roman_empire",
    "amazon_ratings": "amazon_ratings",
    "chameleon_filtered": "chameleon_filtered",
    "chameleon_filtered_directed": "chameleon_filtered_directed",
    "squirrel_filtered": "squirrel_filtered",
    "squirrel_filtered_directed": "squirrel_filtered_directed",
    "tolokers": "tolokers",
    "minesweeper": "minesweeper",
}

SIMPLE_NPZ = {}

LINKX_MAT = {
    "penn94": {
        "data.mat": "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/facebook100/Penn94.mat",
        "fb100-Penn94-splits.npy": "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/fb100-Penn94-splits.npy",
    },
    "reed98": {
        "data.mat": "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/facebook100/Reed98.mat",
    },
    "amherst41": {
        "data.mat": "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/facebook100/Amherst41.mat",
    },
    "cornell5": {
        "data.mat": "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/facebook100/Cornell5.mat",
    },
    "johnshopkins55": {
        "data.mat": "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/facebook100/Johns%20Hopkins55.mat",
    },
}


def _download(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if "graphmining.ai" in url:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl._create_unverified_context()) as src, out.open("wb") as dst:
            dst.write(src.read())
        return
    urllib.request.urlretrieve(url, out)


def _extract_geom_from_dgl_zip(dataset_name: str, out_dir: Path, need_raw: bool):
    zip_url = f"https://data.dgl.ai/dataset/{dataset_name}.zip"
    data = urllib.request.urlopen(zip_url).read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = set(zf.namelist())
        wanted = []
        if need_raw:
            wanted.extend(["out1_node_feature_label.txt", "out1_graph_edges.txt"])
        wanted.extend([name for name in members if name.startswith(f"{dataset_name}_split_0.6_0.2_") and name.endswith(".npz")])
        if not wanted:
            raise FileNotFoundError(f"no compatible files found in {zip_url}")
        for fn in wanted:
            target = out_dir / Path(fn).name
            if target.exists():
                continue
            with zf.open(fn) as src, target.open("wb") as dst:
                dst.write(src.read())


def fetch_geom_gcn(name: str, repo_name: str):
    out_dir = RAW / "node" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/bingzhewei/geom-gcn/master/new_data/{repo_name}"
    need_fallback_raw = False
    for fn in ["out1_node_feature_label.txt", "out1_graph_edges.txt"]:
        dst = out_dir / fn
        if dst.exists():
            continue
        try:
            _download(f"{base}/{fn}", dst)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                need_fallback_raw = True
            else:
                raise
    has_split = any(out_dir.glob(f"{name}_split_0.6_0.2_*.npz"))
    if need_fallback_raw or not has_split:
        _extract_geom_from_dgl_zip(name, out_dir, need_raw=need_fallback_raw)


def fetch_heterophily(name: str, file_stub: str):
    out_dir = RAW / "node" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{file_stub}.npz"
    if not dst.exists():
        _download(f"https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{file_stub}.npz", dst)


def fetch_simple_npz(name: str, url: str):
    out_dir = RAW / "node" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{name}.npz"
    if not dst.exists():
        _download(url, dst)


def fetch_linkx_mat(name: str, files: dict[str, str]):
    out_dir = RAW / "node" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in files.items():
        dst = out_dir / filename
        if not dst.exists():
            _download(url, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["all", "node"], default="all")
    args = ap.parse_args()
    for name, repo_name in GEOM_GCN.items():
        fetch_geom_gcn(name, repo_name)
    for name, file_stub in HETEROPHILY_NPZ.items():
        fetch_heterophily(name, file_stub)
    for name, url in SIMPLE_NPZ.items():
        fetch_simple_npz(name, url)
    for name, files in LINKX_MAT.items():
        fetch_linkx_mat(name, files)
    print("bootstrap finished")


if __name__ == "__main__":
    main()
