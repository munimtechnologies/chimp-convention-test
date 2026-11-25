# SPDX-License-Identifier: Apache-2.0
# MV2, Munim, Inc. Made with AI agents working in MV2's name.
"""Tests for convention_test.py on synthetic data with known answers (numpy + stdlib)."""

import json
import math
import os
import tempfile
import unittest

import numpy as np

import convention_test as ct

TAI_UNITS = [("East", "2019"), ("North", "2019"), ("South", "2019"), ("North-East", "2019")]
TACU_UNITS = [("A Group", "S1AT", "N"), ("A Group", "S2A", "Y"), ("Nursery", "2022", "N"),
              ("Quarantine", "2022", "N"), ("Tom&Nico", "2022", "N")]
SPREAD = ["Groom", "Play", "Rest", "Feed", "Travel"]


def all_units():
    out = [(g, s, "Y", "Tai") for g, s in TAI_UNITS]
    out += [(g, s, lab, "Tacugama") for g, s, lab in TACU_UNITS]
    return out


def write_synth(d):
    long_rows = ["ROLE_MODEL_group;GROUP;SEASON;POPULATION;CONTEXT;AGE"]
    call_rows = ["CALLER;POPULATION;ROLE_MODEL;AGE;AGE_CATEGORIES;CONTEXT;CALL_GROUP;GROUP;SEASON"]
    dyad_rows = ["CALLER1;CALLER2;GROUP;SEASON;ROLE_MODEL;JD;CAT1;CAT2;UTT_CALLER1;UTT_CALLER2;MIN_AGE;GROUP_SIZE"]
    for g, s, lab, pop in all_units():
        long_lab = "With Role Model" if lab == "Y" else "No Role Model"
        if lab == "Y":
            ctxs = ["Groom"] * 20
        else:
            ctxs = [c for c in SPREAD for _ in range(4)]
        for c in ctxs:
            long_rows.append("%s;%s;%s;%s;%s;50" % (long_lab, g, s, pop, c))
        for j in range(1, 5):
            cctx = ["Groom"] * 5 if lab == "Y" else list(SPREAD)
            for c in cctx:
                call_rows.append("ID_%02d;%s;%s;50;[4-5];%s;RS;%s;%s" % (j, pop, lab, c, g, s))
        for j in range(3):
            jd, c1, c2 = ("0,0", "Zero", "High") if lab == "Y" else ("1,0", "One", "Low")
            dyad_rows.append("ID_0%d;ID_0%d;%s;%s;%s;%s;%s;%s;%d;%d;%d,5;10" % (
                j + 1, j + 2, g, s, lab, jd, c1, c2, 3 + j, 5 + 2 * j, 4 + j))
    for name, rows in ((ct.FILE_LONG, long_rows), (ct.FILE_CALLS, call_rows),
                       (ct.FILE_DYADS, dyad_rows)):
        with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")


class TestBasics(unittest.TestCase):
    def test_mi_bits(self):
        self.assertAlmostEqual(ct.mi_bits([[5, 5], [5, 5]]), 0.0, places=12)
        self.assertAlmostEqual(ct.mi_bits([[10, 0], [0, 10]]), 1.0, places=12)

    def test_entropy_bits(self):
        self.assertAlmostEqual(ct.entropy_bits([1, 1, 1, 1]), 2.0, places=12)
        self.assertEqual(ct.entropy_bits([0, 0]), 0.0)

    def test_jsd(self):
        self.assertAlmostEqual(ct.jsd_bits([1, 0], [0, 1]), 1.0, places=12)
        self.assertAlmostEqual(ct.jsd_bits([1, 1], [2, 2]), 0.0, places=12)

    def test_labels_and_numbers(self):
        self.assertEqual(ct.norm_label("With Role Model"), "Y")
        self.assertEqual(ct.norm_label("No role Model\\nTacugama"), "N")
        self.assertEqual(ct.norm_label("No role Model\nTacugama"), "N")
        self.assertEqual(ct.norm_label("Y"), "Y")
        self.assertEqual(ct.norm_label("N"), "N")
        self.assertIsNone(ct.norm_label("maybe"))
        self.assertEqual(ct.parse_num("0,5"), 0.5)
        self.assertIsNone(ct.parse_num("NA"))
        self.assertEqual(ct.population_of("North-East"), "Tai")
        self.assertEqual(ct.population_of("A Group"), "Tacugama")
        self.assertEqual(ct.population_of("x", "Taï"), "Tai")
        self.assertEqual(ct.population_of("x", "Tacugama"), "Tacugama")
        self.assertEqual(ct.find_col(["ROLE.MODEL.group", "B"], "ROLE_MODEL_group"), "ROLE.MODEL.group")

    def test_unit_perm_exhaustive(self):
        stat = ct.make_meandiff_stat([3, 3, 3, 0, 0, 0], 1.0)
        rng = np.random.default_rng(1)
        t = ct.unit_perm_test(stat, 6, [0, 1, 2], rng)
        self.assertTrue(t["exhaustive"])
        self.assertEqual(t["n_assignments"], 20)
        self.assertAlmostEqual(t["p"], 1 / 20, places=12)
        self.assertAlmostEqual(t["min_attainable_p"], 1 / 20, places=12)
        self.assertEqual(t["status"], "OK")
        t2 = ct.unit_perm_test(stat, 6, [0, 1, 2, 3, 4, 5], rng)
        self.assertEqual(t2["status"], "NO_CONTRAST")

    def test_underpowered(self):
        stat = ct.make_meandiff_stat([1, 0, 0, 0, 0], 1.0)
        t = ct.unit_perm_test(stat, 5, [0], np.random.default_rng(2))
        self.assertAlmostEqual(t["p"], 0.2, places=12)
        self.assertEqual(t["status"], "UNDERPOWERED_BY_DESIGN")

    def test_spearman(self):
        self.assertAlmostEqual(ct.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=12)
        self.assertIsNone(ct.spearman([1, 1, 1], [1, 2, 3]))


class TestEndToEnd(unittest.TestCase):
    def test_synthetic_between_sites(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
            write_synth(d)
            res = ct.analyse(d, o, n_perm=200, n_worlds=5, world_perm=50, n_rare=20)
            self.assertEqual(res["n_units"], 9)
            self.assertEqual(res["n_calls"], 180)
            self.assertLessEqual(res["R0_naive_call_mi"]["p"], 0.05)
            self.assertAlmostEqual(res["T1_community_mi"]["p"], 1 / 126, places=12)
            self.assertEqual(res["T1_community_mi"]["n_assignments"], 126)
            t2 = res["T2_entropy"]["rarefied_community"]
            self.assertEqual(t2["rarefy_to_calls"], 20)
            self.assertAlmostEqual(t2["observed"], math.log2(5), places=9)
            self.assertAlmostEqual(t2["p"], 1 / 126, places=12)
            self.assertEqual(res["T4_tacugama"]["n_units"], 5)
            self.assertEqual(res["T4_tacugama"]["mi"]["status"], "UNDERPOWERED_BY_DESIGN")
            self.assertAlmostEqual(res["T4_tacugama"]["mi"]["p"], 0.2, places=12)
            self.assertEqual(res["headline"], "SURVIVES_BETWEEN_SITES_ONLY")
            self.assertEqual(res["T3_dyads"]["n_dyads"], 27)
            self.assertAlmostEqual(res["T3_dyads"]["community"]["p"], 1 / 126, places=12)
            self.assertEqual(res["dyad_reading"], "DYAD_SURVIVES")
            self.assertEqual(res["T5_callers"]["n_callers"], 36)
            self.assertLessEqual(res["T5_callers"]["caller_level"]["p"], 0.05)
            self.assertEqual(len(res["a_group_descriptive"]), 2)
            for f in ("load_report.json", "results.json", "grade.json", "RESULTS-CONVENTION.md"):
                self.assertTrue(os.path.exists(os.path.join(o, f)), f)
            with open(os.path.join(o, "grade.json"), encoding="utf-8") as fh:
                g = json.load(fh)
            self.assertEqual(g["headline"], "SURVIVES_BETWEEN_SITES_ONLY")
            self.assertIsInstance(g["brier_mean"], float)
            outcomes = {r["id"]: r["outcome"] for r in g["predictions"]}
            self.assertTrue(outcomes["P1"])
            self.assertTrue(outcomes["P13"])
            self.assertFalse(outcomes["P14"])
            self.assertTrue(outcomes["P10"])
            self.assertIsNone(outcomes["P12"])

    def test_loader_stop(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
            res = ct.analyse(d, o, n_perm=10, n_worlds=1, world_perm=10, n_rare=5)
            self.assertEqual(res["headline"], "LOADER_STOP")
            with open(os.path.join(o, "grade.json"), encoding="utf-8") as fh:
                g = json.load(fh)
            self.assertFalse(g["predictions"][0]["outcome"])


if __name__ == "__main__":
    unittest.main()
