# PREREG: a community-calibrated test of the chimpanzee "signal convention" (registered 2026-11-24)

MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents working in MV2's name; not yet checked
by a human expert. Text CC BY 4.0, code Apache-2.0.

**Registered 2026-11-24 about 23:40 UTC, BEFORE any data file of the study was opened.** The only
things read before registration were the paper's public descriptions (press releases, the Taï
Chimpanzee Project post), the Zenodo record's file list, and the authors' three public R scripts
(Arbitrariness.R, Social Learning.R, Stability.R; CC BY 4.0). The scripts were read only for column
names, model formulas and factor levels.

## The claim

Valet, Moore, Bortolato, Kleinhause G., ... Crockford (2026), *A signal convention in natural
chimpanzee communication*, PNAS 123(47) e2606837123, https://doi.org/10.1073/pnas.2606837123 .
91 chimpanzees, unvoiced calls (UCs: raspberries, kisses, lip smacks / teeth clacks), four wild
communities in Taï (Côte d'Ivoire) and four orphan peer-raised groups at Tacugama sanctuary (Sierra
Leone). Headline: groups **with adult role models** use UCs in a narrow, stable set of contexts
(mostly grooming), while groups **without** them use UCs across many contexts. One Tacugama group
(A Group) that gained an adult converged on the adult's pattern. Press: "first evidence of learned
vocal conventions in great ape communication".

Data: Zenodo record 22809603, https://doi.org/10.5281/zenodo.22809603 (CC BY 4.0, published
2026-09-17). Downloaded at run time by `get_data.py`, which checks every file against the md5 that
the Zenodo API reports.

## Why a re-test is needed (from the authors' code, not the data)

1. **Mutual information test at the call level.** `Arbitrariness.R` tests the association between
   `ROLE_MODEL_group` and `CONTEXT` with 5,000 permutations of the context labels over single calls.
   Role-model status does not vary between calls; it varies between communities (group-seasons).
   Calls from the same group and the same caller are not independent, so a call-level permutation
   can fire whenever communities differ for any reason.
2. **Entropy model.** Group-season Shannon entropy (plug-in, log2) is regressed on role-model status
   and mean age with no correction for the number of calls; plug-in entropy rises with sample size.
3. **Dyad model.** `Social Learning.R` fits `CAT1 ~ ROLE_MODEL + MIN_AGE.z + GROUP_SIZE.z +
   UTT_CALLER1.z + UTT_CALLER2.z + (1|SEASON) + (1|mm(CALLER1,CALLER2))`: a group-level predictor
   with no group random effect.
4. **Role models are nearly confounded with site.** All Taï groups have adults; the contrast inside
   the sanctuary rests on very few groups.

We found no published reanalysis of this paper (2026-11-24: two web searches, one for
reanalysis / critique / comment / pseudoreplication and one for its methods terms, plus a Zenodo
search; the paper is two weeks old). Scholar, PubPeer and PNAS letters will be checked again
before any result is published.

## Data handling (registered)

- Primary file `Arbitrariness2_anon.csv` (the authors' `data_long`): columns `ROLE_MODEL_group`,
  `GROUP`, `SEASON`, `CONTEXT` required; `POPULATION`, `AGE` optional. Rows with an unknown label or
  empty/NA context are dropped and counted. Labels: text starting "with", or Y/yes, = role model;
  text starting "no"/"without", or N/no, = none.
- **Unit = community-season** = (GROUP, SEASON, label). If a group-season carries both labels it is
  split by label and reported.
- `Arbitrariness_anon.csv` (has `CALLER`) gives caller clusters = (GROUP, CALLER, label).
- `Social learning_anon.csv` (sep ";", dec ",") gives dyads. Overlap score s = 1 - JD if the mean JD
  of rows with CAT1 "Zero" (the authors' "Full overlap") is below that of CAT1 "One"; otherwise s =
  JD. If JD is absent, s = 1 / 0.5 / 0 for CAT1 Zero / Between / One.
- Population: the `POPULATION` column when present, else the group name (Taï: East, North, South,
  North-East; Tacugama: A Group, Nursery, Tom&Nico, Quarantine).
- LOADER_STOP if the primary file is missing or unusable, or gives fewer than 4 units, or lacks
  either label. The loader report (column names, label values, unit sizes) is written either way.

## Tests (seed 20261124, alpha .05, one-sided where a direction is stated)

- **R0 (reproduction):** MI(role model; context) in bits over calls, labels permuted over calls
  (5,000 permutations). Expect p <= .05 as the authors report.
- **T1 (primary):** the same MI, but role-model labels permuted across **whole units**, keeping the
  number of role-model units; exhaustive when C(U,k) <= 200,000, else 100,000 random assignments.
  The minimum attainable p is reported.
- **T2 (primary):** context entropy per unit, plug-in and **rarefied** (units with >= 10 calls, each
  subsampled without replacement to the smallest such unit, 2,000 draws, mean entropy). Statistic:
  mean over no-role-model units minus mean over role-model units; unit-level permutation as in T1.
  Also Spearman(calls per unit, plug-in entropy).
- **T3 (secondary, dyads):** dyad-weighted mean overlap (role model minus none); naive dyad-level
  permutation, then unit-level permutation; also unit means; also overlap residualised on
  log(UTT_CALLER1), log(UTT_CALLER2) and MIN_AGE (OLS).
- **T4 (site confound):** T1 and rarefied T2 restricted to Tacugama units.
- **T5 (intermediate):** MI with labels permuted across callers (all callers; Tacugama callers).
- **Young:** R0, T1 and T2 on calls with AGE < 72 (months, the authors' cut-off).
- **Calibration:** 200 no-effect worlds. The observed unit context profiles are shuffled across the
  unit slots (unit sizes and labels kept) and calls redrawn multinomially. We count how often the
  naive call-level test (500 permutations) and the unit-level test (<= 2,000 assignments) give p <= .05.
- **A Group:** descriptive only (each season's entropy, modal context and Jensen-Shannon distance
  to the pooled Taï profile).

## Headline (registered)

- LOADER_STOP: see above.
- NOT_REPRODUCED: R0 p > .05.
- UNTESTABLE_AT_COMMUNITY_LEVEL: T1's minimum attainable p > .05.
- With A = T1 p <= .05 and B = rarefied T2 p <= .05 with the stated direction (B = A if T2 cannot be
  run):
  - A and B, and T4 MI p <= .05 with a design that can reach .05: **SURVIVES_CLUSTER_CALIBRATION**.
  - A and B otherwise: **SURVIVES_BETWEEN_SITES_ONLY** (the role-model contrast holds across
    communities but cannot be separated from wild vs sanctuary inside this design).
  - neither: **FAILS_CLUSTER_CALIBRATION**.
  - one of the two: MIXED.
- Dyad reading: DYAD_NOT_REPRODUCED / DYAD_SURVIVES / DYAD_FAILS /
  DYAD_UNTESTABLE_AT_COMMUNITY_LEVEL / DYAD_UNAVAILABLE, from T3 naive and unit-level p.

## Forecasts (graded by Brier score; probabilities fixed now)

| id | forecast | p |
|---|---|---|
| P1 | loader passes | .85 |
| P2 | R0 naive call-level MI p <= .05 | .90 |
| P3 | 7 <= U <= 14 units | .55 |
| P4 | T1 community-level MI p <= .05 | .55 |
| P5 | T3 naive dyad-level p <= .05, role model > none | .70 |
| P6 | T3 community-level p <= .05 | .40 |
| P7 | plug-in entropy higher without role models | .85 |
| P8 | rarefied entropy still higher without role models | .80 |
| P9 | rarefied T2 community-level p <= .05 | .45 |
| P10 | within Tacugama, MI is UNDERPOWERED_BY_DESIGN or NO_CONTRAST | .55 |
| P11 | T5 caller-level MI p <= .05 | .80 |
| P12 | Spearman(calls, plug-in entropy) > 0 | .50 |
| P13 | headline SURVIVES_BETWEEN_SITES_ONLY | .35 |
| P14 | headline FAILS_CLUSTER_CALIBRATION | .25 |
| P15 | naive test fires in >= 50% of no-effect worlds | .70 |

## Limits (stated in advance)

- We do not refit the authors' Bayesian models (no R in the job); we test the same contrasts with
  permutation tests at the level where role-model status varies.
- Unit-level permutation treats community-seasons as exchangeable; A Group's two seasons share
  animals. With few units, a true effect can fail to reach .05: a FAILS reading means "not shown at
  the level of communities", not "no convention".
- Wild vs sanctuary differ in far more than adults (habitat, rearing, human care, group age); no
  analysis of these data can separate them, and we say so whatever the numbers are.
- Any change after the data are opened goes in a dated AMENDMENT and cannot change the headline.

## Run

```
python -m unittest -v test_convention_test
python get_data.py --data-dir /data --out-dir /out
python convention_test.py run --data-dir /data --out-dir /out
```
Outputs: download_manifest.json, load_report.json, results.json, grade.json, RESULTS-CONVENTION.md.
