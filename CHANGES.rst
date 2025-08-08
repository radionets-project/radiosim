Radiosim v0.1.1 (2025-08-08)
============================


API Changes
-----------


Bug Fixes
---------


Data Model Changes
------------------


New Features
------------


Maintenance
-----------

- Updated CITATION.cff [`#35 <https://github.com/radionets-project/radiosim/pull/35>`__]

- Updated README

  - Added badges
  - Switched to rst [`#36 <https://github.com/radionets-project/radiosim/pull/36>`__]


Refactoring and Optimization
----------------------------

Radiosim v0.1.0 (2025-08-08)
============================


API Changes
-----------


Bug Fixes
---------


Data Model Changes
------------------


New Features
------------

- Added docs
  - Added changelog CI
  - Added pypi-publish CD
  - Added dependency groups for dev, tests, and docs [`#33 <https://github.com/radionets-project/radiosim/pull/33>`__]


Maintenance
-----------

- Updated `.pre-commit-config.yaml` to use `Ruff <https://docs.astral.sh/ruff>`__

- Updated CI to use py311 and py312

  - Build docs via CI

- Switched to `uv <https://docs.astral.sh/uv>`__ instead of pip for installs in CI
- Added uv lockfile
- Updated docstring formatting
- Switched to `hatch <https://hatch.pypa.io/latest/>`__ as build backend
- Removed unused dependencies and move optional dependencies to the optional dependencies section
- Switched to src layout instead of flat layout, see `*src layout vs flat layout* <https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>`__ for more [`#33 <https://github.com/radionets-project/radiosim/pull/33>`__]


Refactoring and Optimization
----------------------------
