"""Dataset-curation tooling, kept separate from the installable ``basicdet`` wheel.

Two kinds of thing live under this package, and the split is deliberate:

* **Tools** (tracked): :mod:`curation.reid` — cut, review, package, combine and
  audit ReID crops for any tracked-footage source — and
  :mod:`curation.convert_personvit_mim`. A tool is anything still useful for a
  dataset that does not exist yet.
* **Recipes** (``curation/recipes/``, gitignored): one module or YAML per dataset
  actually built here — paths, scenario lists, held-out identities — plus the
  MTMDC detection pipeline and its playbook under ``recipes/mtmdc/``. They are the
  record of this campaign, not part of the framework.
"""
