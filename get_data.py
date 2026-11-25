#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# MV2, Munim, Inc. Made with AI agents working in MV2's name.
"""Download every file of Zenodo record 22809603 (Valet et al. 2026 dataset, CC BY 4.0)
and check each against the md5 and size the Zenodo API reports at run time.
Writes <out-dir>/download_manifest.json. Exit 1 if a needed file is missing or bad."""

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

RECORD = "22809603"
API = "https://zenodo.org/api/records/" + RECORD
NEEDED = ["Arbitrariness2_anon.csv", "Arbitrariness_anon.csv", "Social learning_anon.csv"]
UA = {"User-Agent": "mv2-research-job/1.0 (open research; Apache-2.0)"}


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--out-dir", default="/out")
    a = ap.parse_args(argv)
    dest = os.path.join(a.data_dir, "zenodo-" + RECORD)
    os.makedirs(dest, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    meta = json.loads(fetch(API).decode("utf-8"))
    md = meta.get("metadata", {})
    lic = md.get("license") or {}
    man = {"record": RECORD, "doi": meta.get("doi"), "title": md.get("title"),
           "publication_date": md.get("publication_date"),
           "license": lic.get("id") if isinstance(lic, dict) else lic,
           "access_right": md.get("access_right"), "files": []}
    ok = True
    for f in meta.get("files", []):
        key = f.get("key") or f.get("filename")
        size = f.get("size") or f.get("filesize")
        cks = (f.get("checksum") or "").split(":")[-1]
        url = (f.get("links") or {}).get("self") or (
            API + "/files/" + urllib.parse.quote(key) + "/content")
        if url.endswith("/" + urllib.parse.quote(key)) and "/content" not in url:
            url = url + "/content"
        path = os.path.join(dest, key)
        if not (os.path.exists(path) and md5_of(path) == cks):
            data = fetch(url)
            with open(path, "wb") as fh:
                fh.write(data)
        got = md5_of(path)
        good = (got == cks) and (size is None or os.path.getsize(path) == int(size))
        ok = ok and (good or key not in NEEDED)
        man["files"].append({"key": key, "size": os.path.getsize(path), "api_size": size,
                             "md5": got, "api_md5": cks, "ok": good})
    keys = {x["key"] for x in man["files"]}
    man["missing_needed"] = [k for k in NEEDED if k not in keys]
    ok = ok and not man["missing_needed"]
    man["ok"] = ok
    with open(os.path.join(a.out_dir, "download_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print("download ok" if ok else "DOWNLOAD PROBLEM", json.dumps(man["missing_needed"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
