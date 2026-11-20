# Effect-JEPA

Effect-JEPA publishes the CARLA-Paired Test Set and the CARLA skill used to build paired driving scenes for evaluating action-conditioned world models.

## CARLA-Paired Test Set

Version 1.0 contains 200 evaluation scenes across 12 driving templates, four CARLA towns and seven weather settings. Each scene has a shared 2-second history and five ego action branches with 4-second futures. A sixth replay, `maintain_repeat`, is reserved for quality checks. The release contains 1,000 model branches and 200 QA replays.

Each scene includes six 1600 × 900 RGB cameras and semantic LiDAR sampled at 2 Hz, vehicle and actor state at 20 Hz, action and calibration records, and scene-level QA. Scene IDs run from `scene_0000` through `scene_0199`. The data occupy about 160.47 GiB unpacked.

- [Browse the 200 scenes by template, map and weather](datasets/carla-paired-test/README.md).
- [Download scene 0006](https://github.com/chengjiangzhang/Effect-JEPA/releases/download/carla-paired-test-v1.0/scene_0006.tar.zst).
- [Download the shared maps, split, viewer and manifests](https://github.com/chengjiangzhang/Effect-JEPA/releases/download/carla-paired-test-v1.0/dataset_metadata.tar.zst).

Download all 200 archives, shared files and the checksum list with GNU Wget. You can rerun the command to resume.

```bash
wget --continue --directory-prefix=carla-paired-test --input-file=https://raw.githubusercontent.com/chengjiangzhang/Effect-JEPA/carla-paired-test-v1.0/datasets/carla-paired-test/downloads.txt
```

From the download directory, verify and unpack the release with GNU tar and Zstandard.

```bash
sha256sum --check SHA256SUMS && (for archive in *.tar.zst; do tar --zstd -xf "$archive" || exit 1; done)
```

The scene archives unpack into separate `scene_XXXX/` directories. Unpack the shared archive alongside them so the paths in the split and manifests resolve. Each scene carries its original collection and QA provenance. The `data` field of the published dataset manifest locates each scene relative to the extracted dataset root.

## Collection skill

The skill includes its Python runtime, example configuration and quality criteria. Its workflow describes the CARLA 0.9.16 installation and GPU requirements. The current skill also includes fixes developed after the initial collection. Capture provenance is recorded separately for each scene. [Skill](skills/carla-paired-data/SKILL.md) · [Data contract](skills/carla-paired-data/references/data-contract.md)

The simulations and town assets are from the CARLA project. CARLA credits the Computer Vision Center at the Universitat Autònoma de Barcelona and its contributors. The upstream license section describes the MIT license for CARLA code and CC-BY terms for CARLA assets. [CARLA 0.9.16 acknowledgments and licenses](https://github.com/carla-simulator/carla/tree/0.9.16#licenses) · [CARLA software notice](third_party/CARLA-LICENSE.txt)
