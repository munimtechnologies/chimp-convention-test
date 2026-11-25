#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents working in MV2's name;
# not yet checked by a human expert.
"""chimp-convention-test: a community-calibrated re-test of the chimpanzee
unvoiced-call 'signal convention' (Valet et al. 2026, PNAS 123(47) e2606837123).

Data: Zenodo record 22809603 (CC BY 4.0), downloaded at run time by get_data.py.
Everything here was registered in PREREG.md before any data file was opened; the
column names come only from the authors' public R code.

Usage: python convention_test.py run --data-dir /data --out-dir /out
"""

import argparse
import csv
import io
import itertools
import json
import math
import os
import sys

import numpy as np

SEED = 20261124
N_PERM = 5000
MAX_ENUM = 200000
N_SAMPLE_ASSIGN = 100000
N_RARE_REPS = 2000
RARE_FLOOR = 10
YOUNG_MONTHS = 72.0
N_WORLDS = 200
WORLD_PERM = 500
WORLD_ASSIGN = 2000
ALPHA = 0.05

FILE_LONG = "Arbitrariness2_anon.csv"
FILE_CALLS = "Arbitrariness_anon.csv"
FILE_DYADS = "Social learning_anon.csv"

TACUGAMA_GROUPS = {"agroup", "nursery", "tom&nico", "tomandnico", "quarantine",
                   "ag", "nu", "tn", "qu"}
TAI_GROUPS = {"east", "north", "south", "north-east", "northeast", "n-east",
              "ea", "no", "so", "ne"}

PRIORS = {
    "P1": (0.85, "loader passes (no LOADER_STOP)"),
    "P2": (0.90, "R0: naive call-level MI permutation p <= .05 (reproduces the authors' significant MI)"),
    "P3": (0.55, "number of community-season units U is between 7 and 14"),
    "P4": (0.55, "T1: community-level MI p <= .05"),
    "P5": (0.70, "T3: naive dyad-level overlap difference (role model > none) p <= .05"),
    "P6": (0.40, "T3: community-level dyad overlap p <= .05"),
    "P7": (0.85, "T2: plug-in context entropy higher in no-role-model units than role-model units"),
    "P8": (0.80, "T2: rarefied context entropy still higher in no-role-model units"),
    "P9": (0.45, "T2: rarefied entropy community-level p <= .05"),
    "P10": (0.55, "T4: within Tacugama the MI contrast is UNDERPOWERED_BY_DESIGN or NO_CONTRAST"),
    "P11": (0.80, "T5: caller-level MI permutation p <= .05"),
    "P12": (0.50, "Spearman(calls per unit, plug-in entropy) > 0"),
    "P13": (0.35, "headline is SURVIVES_BETWEEN_SITES_ONLY"),
    "P14": (0.25, "headline is FAILS_CLUSTER_CALIBRATION"),
    "P15": (0.70, "calibration: naive call-level test fires (p <= .05) in >= 50% of no-effect worlds"),
}

NA_STRINGS = {"", "na", "nan", "null", "none"}


# ----------------------------------------------------------------- reading
def col_key(s):
    t = s.strip().strip('"').lstrip("﻿").lower()
    for ch in ". -":
        t = t.replace(ch, "_")
    return "".join(ch for ch in t if ch.isalnum() or ch == "_")


def find_col(header, name):
    want = col_key(name)
    for h in header:
        if col_key(h) == want:
            return h
    return None


def read_table(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    text, enc = "", None
    for e in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(e)
            enc = e
            break
        except UnicodeDecodeError:
            continue
    lines = text.splitlines()
    first = lines[0] if lines else ""
    delim = ";" if first.count(";") >= first.count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    if not rows:
        return {"header": [], "rows": [], "encoding": enc, "delimiter": delim}
    header = [h.strip().strip('"').lstrip("﻿") for h in rows[0]]
    out = []
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        r = list(r) + [""] * max(0, len(header) - len(r))
        out.append({header[i]: r[i].strip() for i in range(len(header))})
    return {"header": header, "rows": out, "encoding": enc, "delimiter": delim}


def parse_num(s):
    if s is None:
        return None
    t = s.strip().replace(",", ".")
    if t.lower() in NA_STRINGS:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def clean_cat(s):
    if s is None:
        return None
    t = " ".join(s.split())
    return None if t.lower() in NA_STRINGS else t


def norm_label(s):
    if s is None:
        return None
    t = " ".join(s.replace("\\n", " ").split()).lower()
    if t in ("y", "yes", "1", "true") or t.startswith("with"):
        return "Y"
    if t in ("n", "no", "0", "false") or t.startswith("no") or t.startswith("without"):
        return "N"
    return None


def group_norm(s):
    return "".join((s or "").strip().lower().split())


def population_of(group, pop_value=None):
    if pop_value:
        t = pop_value.strip().lower()
        if "tacu" in t:
            return "Tacugama"
        if t.startswith("ta"):
            return "Tai"
    g = group_norm(group)
    if g in TACUGAMA_GROUPS:
        return "Tacugama"
    if g in TAI_GROUPS:
        return "Tai"
    return None


def find_file(data_dir, name):
    if not data_dir or not os.path.isdir(data_dir):
        return None
    for root, _, files in os.walk(data_dir):
        if name in files:
            return os.path.join(root, name)
    return None


def _count(d, k):
    d[k] = d.get(k, 0) + 1


# ----------------------------------------------------------------- statistics
def entropy_bits(counts):
    c = np.asarray(counts, dtype=float)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def mi_bits(table):
    t = np.asarray(table, dtype=float)
    n = t.sum()
    if n <= 0:
        return 0.0
    p = t / n
    expected = p.sum(axis=1, keepdims=True) @ p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / expected[nz])).sum())


def jsd_bits(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2.0

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def unit_perm_test(stat_fn, n_units, y_idx, rng, max_enum=MAX_ENUM, n_sample=N_SAMPLE_ASSIGN):
    """Permute the role-model label across whole units (community-seasons), keeping
    the number of role-model units fixed. Exhaustive when C(U,k) <= max_enum."""
    y_idx = sorted(int(i) for i in y_idx)
    k = len(y_idx)
    if n_units < 2 or k == 0 or k == n_units:
        return {"status": "NO_CONTRAST", "n_units": int(n_units), "n_y": k}
    obs = float(stat_fn(frozenset(y_idx)))
    total = math.comb(n_units, k)
    exhaustive = total <= max_enum
    if exhaustive:
        vals = np.array([stat_fn(frozenset(c))
                         for c in itertools.combinations(range(n_units), k)], dtype=float)
    else:
        vals = np.array([stat_fn(frozenset(rng.choice(n_units, k, replace=False).tolist()))
                         for _ in range(n_sample)], dtype=float)
    tol = 1e-9 * max(1.0, abs(obs))
    n_ge = int(np.sum(vals >= obs - tol))
    if exhaustive:
        p = n_ge / len(vals)
        top = float(vals.max())
        min_p = int(np.sum(vals >= top - 1e-9 * max(1.0, abs(top)))) / len(vals)
    else:
        p = (n_ge + 1) / (len(vals) + 1)
        min_p = 1.0 / (len(vals) + 1)
    return {"status": "UNDERPOWERED_BY_DESIGN" if min_p > ALPHA else "OK",
            "n_units": int(n_units), "n_y": k, "n_assignments": int(total),
            "exhaustive": bool(exhaustive), "observed": obs, "p": float(p),
            "min_attainable_p": float(min_p), "null_mean": float(vals.mean())}


def make_mi_stat(unit_counts):
    M = np.asarray(unit_counts, float)
    total = M.sum(axis=0)

    def stat(yset):
        ysum = M[sorted(yset)].sum(axis=0)
        return mi_bits(np.vstack([total - ysum, ysum]))
    return stat


def make_meandiff_stat(values, sign):
    """sign * (mean over role-model units - mean over the others)."""
    v = np.asarray(values, float)
    n = len(v)
    tot = v.sum()

    def stat(yset):
        idx = sorted(yset)
        k = len(idx)
        sy = v[idx].sum()
        return sign * (sy / k - (tot - sy) / (n - k))
    return stat


def make_weighted_diff_stat(sums, counts):
    """dyad-weighted mean(role model) - mean(none), units kept whole."""
    s = np.asarray(sums, float)
    c = np.asarray(counts, float)
    S, C = s.sum(), c.sum()

    def stat(yset):
        idx = sorted(yset)
        sy, cy = s[idx].sum(), c[idx].sum()
        return sy / cy - (S - sy) / (C - cy)
    return stat


def naive_mi_perm(labels, cats, n_cat, rng, n_perm):
    """The authors' kind of test: labels shuffled over single calls."""
    labels = np.asarray(labels, int)
    cats = np.asarray(cats, int)

    def mi_of(lab):
        t = np.bincount(lab * n_cat + cats, minlength=2 * n_cat).reshape(2, n_cat)
        return mi_bits(t)
    obs = mi_of(labels)
    ge = 0
    for _ in range(n_perm):
        if mi_of(rng.permutation(labels)) >= obs - 1e-12:
            ge += 1
    return {"observed": obs, "p": (ge + 1) / (n_perm + 1), "n_perm": int(n_perm),
            "n": int(len(labels))}


def cluster_mi_perm(cluster_counts, cluster_labels, rng, n_perm):
    M = np.asarray(cluster_counts, float)
    lab = np.asarray(cluster_labels, int)
    total = M.sum(axis=0)

    def mi_of(lv):
        ysum = M[lv == 1].sum(axis=0)
        return mi_bits(np.vstack([total - ysum, ysum]))
    if lab.sum() == 0 or lab.sum() == len(lab):
        return {"status": "NO_CONTRAST", "n_clusters": int(len(lab))}
    obs = mi_of(lab)
    ge = 0
    for _ in range(n_perm):
        if mi_of(rng.permutation(lab)) >= obs - 1e-12:
            ge += 1
    return {"status": "OK", "observed": obs, "p": (ge + 1) / (n_perm + 1),
            "n_perm": int(n_perm), "n_clusters": int(len(lab)), "n_y": int(lab.sum())}


def naive_diff_perm(values, labels, rng, n_perm):
    v = np.asarray(values, float)
    lab = np.asarray(labels, int)
    if lab.sum() == 0 or lab.sum() == len(lab):
        return {"status": "NO_CONTRAST"}

    def d(lv):
        return float(v[lv == 1].mean() - v[lv == 0].mean())
    obs = d(lab)
    ge = 0
    for _ in range(n_perm):
        if d(rng.permutation(lab)) >= obs - 1e-12:
            ge += 1
    return {"status": "OK", "observed": obs, "p": (ge + 1) / (n_perm + 1),
            "n_perm": int(n_perm), "n": int(len(v))}


def rarefied_entropy(cats, m, n_cat, rng, reps):
    arr = np.asarray(cats, int)
    if len(arr) == m:
        return entropy_bits(np.bincount(arr, minlength=n_cat))
    hs = np.empty(reps)
    for r in range(reps):
        sub = rng.choice(arr, m, replace=False)
        hs[r] = entropy_bits(np.bincount(sub, minlength=n_cat))
    return float(hs.mean())


# ----------------------------------------------------------------- loaders
def load_long(path):
    info = read_table(path)
    h = info["header"]
    names = ("ROLE_MODEL_group", "GROUP", "SEASON", "CONTEXT", "POPULATION", "AGE")
    cols = {n: find_col(h, n) for n in names}
    rep = {"file": os.path.basename(path), "encoding": info["encoding"],
           "delimiter": info["delimiter"], "header": h, "n_rows": len(info["rows"]),
           "columns_found": cols}
    missing = [n for n in ("ROLE_MODEL_group", "GROUP", "CONTEXT") if cols[n] is None]
    if missing:
        rep["missing"] = missing
        return None, rep
    recs, raw_labels = [], {}
    drop_label = drop_ctx = 0
    for r in info["rows"]:
        raw_lab = r.get(cols["ROLE_MODEL_group"], "")
        _count(raw_labels, raw_lab)
        lab = norm_label(raw_lab)
        ctx = clean_cat(r.get(cols["CONTEXT"], ""))
        if lab is None:
            drop_label += 1
            continue
        if ctx is None:
            drop_ctx += 1
            continue
        grp = clean_cat(r.get(cols["GROUP"], "")) or "?"
        season = (clean_cat(r.get(cols["SEASON"], "")) if cols["SEASON"] else "") or ""
        popv = r.get(cols["POPULATION"], "") if cols["POPULATION"] else ""
        age = parse_num(r.get(cols["AGE"], "")) if cols["AGE"] else None
        recs.append({"label": lab, "group": grp, "season": season, "context": ctx,
                     "pop": population_of(grp, popv), "age": age})
    rep.update({"raw_label_values": raw_labels, "dropped_unknown_label": drop_label,
                "dropped_missing_context": drop_ctx, "n_used": len(recs),
                "n_age_parsed": sum(1 for r in recs if r["age"] is not None)})
    return recs, rep


def load_calls(path):
    info = read_table(path)
    h = info["header"]
    names = ("CALLER", "ROLE_MODEL", "CONTEXT", "GROUP", "SEASON", "AGE", "POPULATION")
    cols = {n: find_col(h, n) for n in names}
    rep = {"file": os.path.basename(path), "encoding": info["encoding"],
           "delimiter": info["delimiter"], "header": h, "n_rows": len(info["rows"]),
           "columns_found": cols}
    missing = [n for n in ("CALLER", "ROLE_MODEL", "CONTEXT", "GROUP") if cols[n] is None]
    if missing:
        rep["missing"] = missing
        return None, rep
    recs, raw_labels = [], {}
    dropped = 0
    for r in info["rows"]:
        raw_lab = r.get(cols["ROLE_MODEL"], "")
        _count(raw_labels, raw_lab)
        lab = norm_label(raw_lab)
        ctx = clean_cat(r.get(cols["CONTEXT"], ""))
        caller = clean_cat(r.get(cols["CALLER"], ""))
        if lab is None or ctx is None or caller is None:
            dropped += 1
            continue
        grp = clean_cat(r.get(cols["GROUP"], "")) or "?"
        popv = r.get(cols["POPULATION"], "") if cols["POPULATION"] else ""
        recs.append({"label": lab, "group": grp, "caller": caller, "context": ctx,
                     "pop": population_of(grp, popv)})
    rep.update({"raw_label_values": raw_labels, "dropped": dropped, "n_used": len(recs)})
    return recs, rep


def load_dyads(path):
    info = read_table(path)
    h = info["header"]
    names = ("GROUP", "SEASON", "ROLE_MODEL", "JD", "CAT1", "UTT_CALLER1", "UTT_CALLER2",
             "MIN_AGE", "CALLER1", "CALLER2")
    cols = {n: find_col(h, n) for n in names}
    rep = {"file": os.path.basename(path), "encoding": info["encoding"],
           "delimiter": info["delimiter"], "header": h, "n_rows": len(info["rows"]),
           "columns_found": cols}
    if cols["GROUP"] is None or cols["ROLE_MODEL"] is None or (cols["JD"] is None and cols["CAT1"] is None):
        rep["missing"] = [n for n in ("GROUP", "ROLE_MODEL", "JD|CAT1")
                          if (n == "JD|CAT1" and cols["JD"] is None and cols["CAT1"] is None)
                          or (n != "JD|CAT1" and cols[n] is None)]
        return None, rep
    rows, raw_labels, raw_cat1 = [], {}, {}
    for r in info["rows"]:
        raw_lab = r.get(cols["ROLE_MODEL"], "")
        _count(raw_labels, raw_lab)
        cat1 = clean_cat(r.get(cols["CAT1"], "")) if cols["CAT1"] else None
        _count(raw_cat1, cat1 or "")
        rows.append({"label": norm_label(raw_lab),
                     "group": clean_cat(r.get(cols["GROUP"], "")) or "?",
                     "season": (clean_cat(r.get(cols["SEASON"], "")) if cols["SEASON"] else "") or "",
                     "jd": parse_num(r.get(cols["JD"], "")) if cols["JD"] else None,
                     "cat1": cat1.lower() if cat1 else None,
                     "utt1": parse_num(r.get(cols["UTT_CALLER1"], "")) if cols["UTT_CALLER1"] else None,
                     "utt2": parse_num(r.get(cols["UTT_CALLER2"], "")) if cols["UTT_CALLER2"] else None,
                     "min_age": parse_num(r.get(cols["MIN_AGE"], "")) if cols["MIN_AGE"] else None})
    # direction rule (registered): CAT1 'Zero' is the authors' 'Full overlap'.
    zero = [x["jd"] for x in rows if x["cat1"] == "zero" and x["jd"] is not None]
    one = [x["jd"] for x in rows if x["cat1"] == "one" and x["jd"] is not None]
    if cols["JD"] is not None:
        if zero and one:
            direction = "distance" if np.mean(zero) < np.mean(one) else "similarity"
        else:
            direction = "distance_assumed"
        for x in rows:
            if x["jd"] is None:
                x["s"] = None
            else:
                x["s"] = (1.0 - x["jd"]) if direction.startswith("distance") else x["jd"]
        score_source = "JD"
    else:
        direction = "cat1_score"
        cmap = {"zero": 1.0, "between": 0.5, "one": 0.0}
        for x in rows:
            x["s"] = cmap.get(x["cat1"])
        score_source = "CAT1"
    used = [x for x in rows if x["label"] is not None and x["s"] is not None]
    for x in used:
        x["pop"] = population_of(x["group"])
    rep.update({"raw_label_values": raw_labels, "raw_cat1_values": raw_cat1,
                "jd_direction": direction, "score_source": score_source,
                "mean_jd_cat1_zero": float(np.mean(zero)) if zero else None,
                "mean_jd_cat1_one": float(np.mean(one)) if one else None,
                "n_used": len(used), "dropped": len(rows) - len(used)})
    return used, rep


# ----------------------------------------------------------------- units
def build_units(recs, contexts, keyfn):
    cidx = {c: i for i, c in enumerate(contexts)}
    units = {}
    for r in recs:
        key = keyfn(r)
        u = units.get(key)
        if u is None:
            u = {"key": key, "group": r["group"], "season": r.get("season", ""),
                 "label": r["label"], "pop": r.get("pop"),
                 "counts": np.zeros(len(contexts)), "cats": []}
            units[key] = u
        u["counts"][cidx[r["context"]]] += 1
        u["cats"].append(cidx[r["context"]])
    return [units[k] for k in sorted(units)]


def unit_key(r):
    return (r["group"], r.get("season", ""), r["label"])


def y_positions(units):
    return [i for i, u in enumerate(units) if u["label"] == "Y"]


def unit_summary(u, contexts):
    n = int(u["counts"].sum())
    top = int(np.argmax(u["counts"]))
    return {"group": u["group"], "season": u["season"], "label": u["label"],
            "population": u["pop"], "n_calls": n,
            "entropy_plugin_bits": entropy_bits(u["counts"]),
            "modal_context": contexts[top],
            "modal_share": float(u["counts"][top] / n) if n else None}


def mi_tests(units, contexts, rng, n_perm, with_naive=True):
    out = {}
    if with_naive:
        labels, cats = [], []
        for u in units:
            labels.extend([1 if u["label"] == "Y" else 0] * len(u["cats"]))
            cats.extend(u["cats"])
        if 0 < sum(labels) < len(labels):
            out["naive"] = naive_mi_perm(labels, cats, len(contexts), rng, n_perm)
        else:
            out["naive"] = {"status": "NO_CONTRAST"}
    out["community"] = unit_perm_test(make_mi_stat([u["counts"] for u in units]),
                                      len(units), y_positions(units), rng)
    return out


def entropy_tests(units, contexts, rng, n_rare):
    res = {}
    H = [entropy_bits(u["counts"]) for u in units]
    ns = [int(u["counts"].sum()) for u in units]
    res["spearman_n_vs_plugin_entropy"] = spearman(ns, H)
    res["plugin_community"] = unit_perm_test(make_meandiff_stat(H, -1.0), len(units),
                                             y_positions(units), rng)
    keep = [i for i, n in enumerate(ns) if n >= RARE_FLOOR]
    if not keep:
        res["rarefied_community"] = {"status": "NO_UNITS"}
        return res
    m = min(ns[i] for i in keep)
    rH = [rarefied_entropy(units[i]["cats"], m, len(contexts), rng, n_rare) for i in keep]
    kept_units = [units[i] for i in keep]
    t = unit_perm_test(make_meandiff_stat(rH, -1.0), len(kept_units),
                       y_positions(kept_units), rng)
    t.update({"rarefy_to_calls": int(m), "n_units_excluded_below_floor": len(units) - len(keep),
              "rarefied_entropy_by_unit": [
                  {"group": u["group"], "season": u["season"], "label": u["label"], "H": h}
                  for u, h in zip(kept_units, rH)]})
    res["rarefied_community"] = t
    return res


def null_worlds(units, contexts, rng, n_worlds, world_perm):
    """No-effect worlds: the observed unit context profiles are shuffled across the
    unit slots (sizes and labels kept), and calls are redrawn multinomially."""
    U = len(units)
    n_cat = len(contexts)
    sizes = [int(u["counts"].sum()) for u in units]
    profiles = [u["counts"] / u["counts"].sum() for u in units]
    lab_units = [1 if u["label"] == "Y" else 0 for u in units]
    y_idx = [i for i in range(U) if lab_units[i] == 1]
    naive_hits = comm_hits = 0
    for _ in range(n_worlds):
        perm = rng.permutation(U)
        counts = np.array([rng.multinomial(sizes[i], profiles[perm[i]]) for i in range(U)])
        labels = np.concatenate([np.full(sizes[i], lab_units[i]) for i in range(U)])
        cats = np.concatenate([np.repeat(np.arange(n_cat), counts[i]) for i in range(U)])
        nv = naive_mi_perm(labels, cats, n_cat, rng, world_perm)
        if nv["p"] <= ALPHA:
            naive_hits += 1
        ct = unit_perm_test(make_mi_stat(counts), U, y_idx, rng,
                            max_enum=WORLD_ASSIGN, n_sample=WORLD_ASSIGN)
        if ct.get("p", 1.0) <= ALPHA:
            comm_hits += 1
    return {"n_worlds": int(n_worlds), "naive_hits": naive_hits,
            "naive_false_positive_rate": naive_hits / n_worlds if n_worlds else None,
            "community_hits": comm_hits,
            "community_false_positive_rate": comm_hits / n_worlds if n_worlds else None}


# ----------------------------------------------------------------- readings
def status_of(t):
    if not t or t.get("status") in (None, "NO_CONTRAST", "NO_UNITS", "UNAVAILABLE"):
        return (t or {}).get("status", "UNAVAILABLE")
    return t["status"]


def headline_of(res):
    if res.get("loader_stop"):
        return "LOADER_STOP"
    r0 = res["R0_naive_call_mi"]
    if r0.get("status") == "NO_CONTRAST" or r0["p"] > ALPHA:
        return "NOT_REPRODUCED"
    t1 = res["T1_community_mi"]
    if t1["status"] == "NO_CONTRAST" or t1["min_attainable_p"] > ALPHA:
        return "UNTESTABLE_AT_COMMUNITY_LEVEL"
    a = t1["p"] <= ALPHA
    t2 = res.get("T2_entropy", {}).get("rarefied_community", {})
    if "p" in t2:
        b = t2["p"] <= ALPHA and t2["observed"] > 0
    else:
        b = a
    if a and b:
        w = res.get("T4_tacugama", {}).get("mi", {})
        if w.get("status") == "OK" and w.get("p", 1.0) <= ALPHA:
            return "SURVIVES_CLUSTER_CALIBRATION"
        return "SURVIVES_BETWEEN_SITES_ONLY"
    if not a and not b:
        return "FAILS_CLUSTER_CALIBRATION"
    return "MIXED"


def dyad_reading(t3):
    if not t3 or "naive" not in t3:
        return "DYAD_UNAVAILABLE"
    nv = t3["naive"]
    if nv.get("status") != "OK" or nv["p"] > ALPHA or nv["observed"] <= 0:
        return "DYAD_NOT_REPRODUCED"
    c = t3["community"]
    if c.get("status") == "NO_CONTRAST":
        return "DYAD_UNAVAILABLE"
    if c["min_attainable_p"] > ALPHA:
        return "DYAD_UNTESTABLE_AT_COMMUNITY_LEVEL"
    return "DYAD_SURVIVES" if c["p"] <= ALPHA else "DYAD_FAILS"


def _p_le(t):
    if not t or "p" not in t:
        return None
    return bool(t["p"] <= ALPHA)


def evaluate(res):
    hl = res["headline"]
    o = {k: None for k in PRIORS}
    o["P1"] = hl != "LOADER_STOP"
    if hl == "LOADER_STOP":
        return o
    o["P2"] = _p_le(res["R0_naive_call_mi"])
    o["P3"] = 7 <= res["n_units"] <= 14
    o["P4"] = _p_le(res["T1_community_mi"])
    t3 = res.get("T3_dyads", {})
    nv = t3.get("naive", {})
    if "p" in nv:
        o["P5"] = bool(nv["p"] <= ALPHA and nv["observed"] > 0)
    o["P6"] = _p_le(t3.get("community"))
    t2 = res.get("T2_entropy", {})
    if "observed" in t2.get("plugin_community", {}):
        o["P7"] = t2["plugin_community"]["observed"] > 0
    if "observed" in t2.get("rarefied_community", {}):
        o["P8"] = t2["rarefied_community"]["observed"] > 0
    o["P9"] = _p_le(t2.get("rarefied_community"))
    w = res.get("T4_tacugama", {}).get("mi")
    if w:
        o["P10"] = w.get("status") in ("UNDERPOWERED_BY_DESIGN", "NO_CONTRAST")
    o["P11"] = _p_le(res.get("T5_callers", {}).get("caller_level"))
    sp = t2.get("spearman_n_vs_plugin_entropy")
    if sp is not None:
        o["P12"] = sp > 0
    o["P13"] = hl == "SURVIVES_BETWEEN_SITES_ONLY"
    o["P14"] = hl == "FAILS_CLUSTER_CALIBRATION"
    cal = res.get("calibration", {})
    if cal.get("naive_false_positive_rate") is not None:
        o["P15"] = cal["naive_false_positive_rate"] >= 0.5
    return o


def grade(res):
    o = evaluate(res)
    rows, sq = [], []
    for k, (prior, text) in PRIORS.items():
        out = o[k]
        b = None if out is None else (prior - (1.0 if out else 0.0)) ** 2
        if b is not None:
            sq.append(b)
        rows.append({"id": k, "prediction": text, "prior": prior,
                     "outcome": out, "brier": b})
    return {"headline": res["headline"], "dyad_reading": res.get("dyad_reading"),
            "predictions": rows, "n_graded": len(sq),
            "brier_mean": float(np.mean(sq)) if sq else None}


# ----------------------------------------------------------------- main analysis
def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(str(type(o)))


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=_json_default)


def analyse(data_dir, out_dir, n_perm=N_PERM, n_worlds=N_WORLDS, world_perm=WORLD_PERM,
            n_rare=N_RARE_REPS):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    load_report = {"files": {}}
    res = {"seed": SEED, "alpha": ALPHA}

    p_long = find_file(data_dir, FILE_LONG)
    recs, rep = (None, {"missing_file": FILE_LONG}) if p_long is None else load_long(p_long)
    load_report["files"][FILE_LONG] = rep
    units = []
    if recs:
        contexts = sorted({r["context"] for r in recs})
        units = build_units(recs, contexts, unit_key)
        mixed = {}
        for u in units:
            _count(mixed, (u["group"], u["season"]))
        load_report["group_seasons_with_both_labels"] = [list(k) for k, v in mixed.items() if v > 1]
        load_report["contexts"] = contexts
        load_report["units"] = [{"group": u["group"], "season": u["season"], "label": u["label"],
                                 "population": u["pop"], "n_calls": int(u["counts"].sum())}
                                for u in units]
    labels_present = {u["label"] for u in units}
    if not recs or len(units) < 4 or labels_present != {"Y", "N"}:
        load_report["loader_stop_reason"] = ("no usable rows" if not recs else
                                             "fewer than 4 units or one label missing")
        res.update({"loader_stop": True, "n_units": len(units)})
        res["headline"] = headline_of(res)
        _dump(os.path.join(out_dir, "load_report.json"), load_report)
        _dump(os.path.join(out_dir, "results.json"), res)
        g = grade(res)
        _dump(os.path.join(out_dir, "grade.json"), g)
        return res

    res["n_units"] = len(units)
    res["n_calls"] = int(sum(u["counts"].sum() for u in units))
    res["units"] = [unit_summary(u, contexts) for u in units]

    # R0 + T1: MI between role-model status and context
    mt = mi_tests(units, contexts, rng, n_perm)
    res["R0_naive_call_mi"] = mt["naive"]
    res["T1_community_mi"] = mt["community"]

    # T2: context entropy, plug-in and rarefied
    res["T2_entropy"] = entropy_tests(units, contexts, rng, n_rare)

    # Young (< 72 months) subset
    young = [r for r in recs if r["age"] is not None and r["age"] < YOUNG_MONTHS]
    if young:
        yu = build_units(young, contexts, unit_key)
        if {u["label"] for u in yu} == {"Y", "N"}:
            ymt = mi_tests(yu, contexts, rng, n_perm)
            res["young"] = {"n_calls": len(young), "n_units": len(yu),
                            "naive_mi": ymt["naive"], "community_mi": ymt["community"],
                            "entropy": entropy_tests(yu, contexts, rng, n_rare)}
        else:
            res["young"] = {"status": "NO_CONTRAST", "n_calls": len(young)}
    else:
        res["young"] = {"status": "UNAVAILABLE"}

    # T4: within Tacugama only
    tu = [u for u in units if u["pop"] == "Tacugama"]
    t4 = {"n_units": len(tu), "n_y": sum(1 for u in tu if u["label"] == "Y")}
    if len(tu) >= 2:
        t4["mi"] = mi_tests(tu, contexts, rng, n_perm, with_naive=False)["community"]
        t4["mi_naive"] = mi_tests(tu, contexts, rng, n_perm)["naive"]
        t4["entropy"] = entropy_tests(tu, contexts, rng, n_rare)["rarefied_community"]
    else:
        t4["mi"] = {"status": "NO_CONTRAST"}
    res["T4_tacugama"] = t4

    # A Group descriptive (the group that gained an adult)
    tai_y = [u for u in units if u["pop"] == "Tai" and u["label"] == "Y"]
    tai_prof = np.sum([u["counts"] for u in tai_y], axis=0) if tai_y else None
    ag = []
    for u in units:
        if group_norm(u["group"]) in ("agroup", "ag"):
            d = unit_summary(u, contexts)
            if tai_prof is not None and tai_prof.sum() > 0:
                d["jsd_to_tai_pooled_bits"] = jsd_bits(u["counts"], tai_prof)
            ag.append(d)
    res["a_group_descriptive"] = ag

    # T5: caller-level clusters from Arbitrariness_anon.csv
    p_calls = find_file(data_dir, FILE_CALLS)
    crecs, crep = (None, {"missing_file": FILE_CALLS}) if p_calls is None else load_calls(p_calls)
    load_report["files"][FILE_CALLS] = crep
    if crecs:
        cctx = sorted({r["context"] for r in crecs})
        callers = build_units(crecs, cctx, lambda r: (r["group"], r["caller"], r["label"]))
        clab = [1 if c["label"] == "Y" else 0 for c in callers]
        t5 = {"n_callers": len(callers), "n_calls": len(crecs),
              "caller_level": cluster_mi_perm([c["counts"] for c in callers], clab, rng, n_perm)}
        labels, cats = [], []
        for c in callers:
            labels.extend([1 if c["label"] == "Y" else 0] * len(c["cats"]))
            cats.extend(c["cats"])
        t5["naive_call_level"] = (naive_mi_perm(labels, cats, len(cctx), rng, n_perm)
                                  if 0 < sum(labels) < len(labels) else {"status": "NO_CONTRAST"})
        tc = [c for c in callers if c["pop"] == "Tacugama"]
        t5["tacugama_caller_level"] = cluster_mi_perm(
            [c["counts"] for c in tc], [1 if c["label"] == "Y" else 0 for c in tc], rng, n_perm
        ) if tc else {"status": "NO_CONTRAST"}
        res["T5_callers"] = t5
    else:
        res["T5_callers"] = {"status": "UNAVAILABLE"}

    # T3: dyadic context overlap (Social learning_anon.csv)
    p_dy = find_file(data_dir, FILE_DYADS)
    drecs, drep = (None, {"missing_file": FILE_DYADS}) if p_dy is None else load_dyads(p_dy)
    load_report["files"][FILE_DYADS] = drep
    if drecs:
        t3 = {"n_dyads": len(drecs)}
        s = [x["s"] for x in drecs]
        lab = [1 if x["label"] == "Y" else 0 for x in drecs]
        t3["naive"] = naive_diff_perm(s, lab, rng, n_perm)
        dunits = {}
        for x in drecs:
            k = (x["group"], x["season"], x["label"])
            d = dunits.setdefault(k, {"label": x["label"], "pop": x["pop"], "sum": 0.0, "n": 0})
            d["sum"] += x["s"]
            d["n"] += 1
        dk = sorted(dunits)
        dl = [dunits[k] for k in dk]
        yi = [i for i, d in enumerate(dl) if d["label"] == "Y"]
        t3["n_units"] = len(dl)
        t3["units"] = [{"group": k[0], "season": k[1], "label": k[2], "n_dyads": dunits[k]["n"],
                        "mean_overlap": dunits[k]["sum"] / dunits[k]["n"]} for k in dk]
        t3["community"] = unit_perm_test(make_weighted_diff_stat([d["sum"] for d in dl],
                                                                 [d["n"] for d in dl]),
                                         len(dl), yi, rng)
        t3["community_unit_means"] = unit_perm_test(
            make_meandiff_stat([d["sum"] / d["n"] for d in dl], 1.0), len(dl), yi, rng)
        ti = [i for i, d in enumerate(dl) if d["pop"] == "Tacugama"]
        if len(ti) >= 2:
            sub = [dl[i] for i in ti]
            t3["tacugama"] = unit_perm_test(
                make_weighted_diff_stat([d["sum"] for d in sub], [d["n"] for d in sub]),
                len(sub), [j for j, d in enumerate(sub) if d["label"] == "Y"], rng)
        else:
            t3["tacugama"] = {"status": "NO_CONTRAST"}
        # secondary: overlap adjusted for utterance counts (and minimum age)
        adj = [x for x in drecs if x["utt1"] and x["utt2"] and x["utt1"] > 0 and x["utt2"] > 0]
        if len(adj) >= 10:
            use_age = sum(1 for x in adj if x["min_age"] is not None) >= 0.95 * len(adj)
            if use_age:
                adj = [x for x in adj if x["min_age"] is not None]
            X = np.array([[1.0, math.log(x["utt1"]), math.log(x["utt2"])]
                          + ([x["min_age"]] if use_age else []) for x in adj])
            yv = np.array([x["s"] for x in adj])
            beta = np.linalg.lstsq(X, yv, rcond=None)[0]
            r = yv - X @ beta
            alab = [1 if x["label"] == "Y" else 0 for x in adj]
            au = {}
            for x, ri in zip(adj, r):
                k = (x["group"], x["season"], x["label"])
                d = au.setdefault(k, {"label": x["label"], "sum": 0.0, "n": 0})
                d["sum"] += float(ri)
                d["n"] += 1
            al = [au[k] for k in sorted(au)]
            t3["adjusted"] = {"n": len(adj), "uses_min_age": bool(use_age),
                              "naive": naive_diff_perm(r, alab, rng, n_perm),
                              "community": unit_perm_test(
                                  make_weighted_diff_stat([d["sum"] for d in al], [d["n"] for d in al]),
                                  len(al), [i for i, d in enumerate(al) if d["label"] == "Y"], rng)}
        res["T3_dyads"] = t3
    else:
        res["T3_dyads"] = {"status": "UNAVAILABLE"}
    res["dyad_reading"] = dyad_reading(res.get("T3_dyads"))

    # calibration: how often do the two tests fire in no-effect worlds?
    res["calibration"] = null_worlds(units, contexts, rng, n_worlds, world_perm)

    res["headline"] = headline_of(res)
    g = grade(res)
    _dump(os.path.join(out_dir, "load_report.json"), load_report)
    _dump(os.path.join(out_dir, "results.json"), res)
    _dump(os.path.join(out_dir, "grade.json"), g)
    write_markdown(res, g, os.path.join(out_dir, "RESULTS-CONVENTION.md"))
    return res


def _fmt(t, key="p"):
    if not t or key not in t:
        return (t or {}).get("status", "n/a")
    return "%.4g" % t[key]


def write_markdown(res, g, path):
    t1 = res["T1_community_mi"]
    t2 = res["T2_entropy"]["rarefied_community"]
    lines = [
        "# RESULTS-CONVENTION (machine-written by convention_test.py)",
        "",
        "Headline: %s. Dyad reading: %s." % (res["headline"], res["dyad_reading"]),
        "",
        "| Test | Statistic | p |",
        "|---|---|---|",
        "| R0 naive call-level MI | %s bits | %s |" % (_fmt(res["R0_naive_call_mi"], "observed"),
                                                    _fmt(res["R0_naive_call_mi"])),
        "| T1 community-level MI | %s bits | %s (min attainable %s) |" % (
            _fmt(t1, "observed"), _fmt(t1), _fmt(t1, "min_attainable_p")),
        "| T2 rarefied entropy, none minus role model | %s bits | %s |" % (_fmt(t2, "observed"), _fmt(t2)),
        "| T4 within Tacugama MI | %s | %s |" % (res["T4_tacugama"].get("mi", {}).get("status", "n/a"),
                                              _fmt(res["T4_tacugama"].get("mi"))),
        "",
        "Units: %d community-seasons, %d calls. Brier (mean over %d graded forecasts): %s." % (
            res["n_units"], res["n_calls"], g["n_graded"],
            "%.4f" % g["brier_mean"] if g["brier_mean"] is not None else "n/a"),
        "",
        "Calibration: naive call-level test fired in %s of %s no-effect worlds; community test in %s." % (
            res["calibration"]["naive_hits"], res["calibration"]["n_worlds"],
            res["calibration"]["community_hits"]),
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    res = analyse(args.data_dir, args.out_dir)
    print(res["headline"], res.get("dyad_reading", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
