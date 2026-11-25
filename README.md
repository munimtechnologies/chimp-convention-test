# chimp-convention-test

A pre-registered, community-calibrated re-test of the 2026 PNAS claim that chimpanzee unvoiced
calls ("raspberries") are a **signal convention** learned from adult role models
(Valet et al. 2026, https://doi.org/10.1073/pnas.2606837123).

The question: the published evidence comes from tests that treat single calls or dyads as
independent, but role-model status varies only between communities, and it is almost fully
confounded with wild Taï vs sanctuary Tacugama. Does the contrast survive when it is tested at the
level where it varies, and can this design separate role models from site at all?

- **PREREG.md**: the plan, verdict rules and 15 forecasts, registered before any data was opened.
- **convention_test.py**: the analysis (Python + numpy). **get_data.py** downloads the authors'
  public data (Zenodo 22809603, CC BY 4.0) and checks md5s. **test_convention_test.py**: tests on
  synthetic data with known answers.
- Results will appear in `results/` with a reading, once the job has run.

Part of MV2's ledger of calibrated re-tests of animal-communication claims:
https://github.com/munimtechnologies/calibrated-animal-claims

MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents working in MV2's name; not yet checked
by a human expert. Code Apache-2.0; text CC BY 4.0. The data belong to their authors (CC BY 4.0)
and are not redistributed here.
