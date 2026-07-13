# Fork provenance

This public fork is based on the upstream MetaDrive commit
[`85e5dadc6c7436d324348f6e3d8f8e680c06b4db`](https://github.com/metadriverse/metadrive/commit/85e5dadc6c7436d324348f6e3d8f8e680c06b4db).

It contains only the runtime changes required by the accompanying cognitive-driving experiments:

- initialize IDM traffic target speeds in the 60–80 km/h range and retain an 80 km/h cap;
- guard Panda3D cleanup against already-invalidated `NodePath` objects.

No experiment data, logs, checkpoints, private project code, or MetaDrive asset archives are stored in this fork. MetaDrive's existing `pull_asset.py` continues to download the official `MetaDrive-0.4.3` assets from the upstream release.

MetaDrive remains licensed under Apache-2.0; see `LICENSE.txt`.
