# Dependencies and provenance

Runtime dependencies are installed from their own distributions: NumPy and PyYAML, plus optional PyArrow. No dependency source tree, wheel, virtual environment or executable is bundled. Testing uses pytest. Dependency licenses remain with their respective projects.

The repository contains the authors' research implementation of conditional shortcut optimization, geometry/relationship tests, repair and event-index controls. Classical shortcut-path and synchronous-error simplification ideas are identified by the method descriptions; no external solver is required.

The historical tables include planar SQUISH-E and CISED-S adaptations. Their implementation file and the separately checked-out MLSimp repository are not redistributed in this snapshot. The original source references are:

- [Trajectory-compression experimental study repository](https://github.com/uestc-db/traj-compression), referenced during SQUISH-E adaptation.
- Lin et al., VLDB Journal (2019), [doi:10.1007/s00778-019-00575-8](https://doi.org/10.1007/s00778-019-00575-8), referenced during CISED-S adaptation.

Those four external native/repaired rows remain labeled as recorded historical comparisons. They are not advertised as executable methods in this release. Dataset readers implement only file-format access and do not vendor dataset toolkits.
