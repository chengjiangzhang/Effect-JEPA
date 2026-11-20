"""Offline publication tests: real curation, tiny synthetic sealed captures, no CARLA."""
import copy
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import curate_dataset as curate
import publish_dataset as publish
from test_curation import fixture


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.work = self.base / 'work'
        self.work.mkdir()
        self.output = self.base / 'dataset'
        self.entries = []
        self.manifest = self.work / 'dataset_manifest.json'
        self.curation = self.base / 'curation/curation_manifest.json'

    def add_scene(self, scene='scene_0000', delta=0.):
        scratch = self.base / ('fixture_' + scene)
        entry, spec, states, execution = fixture(scratch, scene, delta=delta)
        attempt = self.work / 'attempts' / scene / 'candidate_00/attempt_01'
        data, reports = attempt / 'data', attempt / 'reports'
        data.mkdir(parents=True)
        reports.mkdir()
        folder = data / scene
        shutil.move(str(scratch / 'captures' / scene), folder)
        spec.update(map_geometry_path='maps/Town04.xodr', render_preset={'file': 'render_preset.json'},
                    semantic_lidar={'name': 'LIDAR_TOP'})
        publish.write(folder / 'spec.json', spec)
        execution['spec_file_sha256'] = publish.file_sha(folder / 'spec.json')
        publish.write(reports / 'execution.json', {'scenes': {scene: execution}})
        for name in ('protocol', 'render', 'equivalence', 'physics', 'action_execution', 'action_selection', 'physics_execution'):
            publish.write(reports / (name + '.json'), {'status': 'passed'})
        (data / 'maps').mkdir()
        (data / 'maps/Town04.xodr').write_text('<OpenDRIVE/>')
        publish.write(data / 'render_preset.json', {'preset': 'fixture'})
        publish.write(data / 'manifest.json', {'version': 'paired-multicam-v2', 'scene_ids': [scene],
            'scene_branches': {scene: spec['branches']}, 'qa_only_branches': ['maintain_repeat']})
        common = attempt / 'export' / scene
        publish.write(common / 'frame.json', {'reference_branch': 'maintain', 'model_branches': entry['model_branches'],
            'qa_only_branches': ['maintain_repeat']})
        publish.write(common / 'action_selection.json', {'similar_motion_branches': {}})
        for branch in spec['branches']:
            b = folder / 'branches' / branch
            b.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(b / 'states.npz', **states)
            np.savez_compressed(b / 'samples.npz', time_s=states['time_s'])
            np.savez_compressed(b / 'pedestrians.npz', time_s=states['time_s'])
            publish.write(b / 'meta.json', {'status': 'complete', 'camera_names': ['CAM_FRONT'], 'collisions': []})
            (b / 'camera/CAM_FRONT').mkdir(parents=True)
            (b / 'lidar/LIDAR_TOP').mkdir(parents=True)
            for i in range(3):
                (b / f'camera/CAM_FRONT/{i:05d}.png').write_bytes(b'fixture-not-a-decoded-image' + bytes([i]))
                np.savez_compressed(b / f'lidar/LIDAR_TOP/{i:05d}.npz', x=np.zeros(1))
            dest = common / branch
            publish.write(dest / 'lidar.json', {'raw_files': [os.path.relpath(b / f'lidar/LIDAR_TOP/{i:05d}.npz', dest) for i in range(3)]})
            publish.write(dest / 'pedestrian_tracks.json', {'raw_record': os.path.relpath(b / 'pedestrians.npz', dest)})
            np.savez_compressed(dest / 'states_ego0.npz', position=states['position'])
        publish.write(attempt / 'common_history_000/metadata.json', {'purpose': 'observed prefix evidence'})
        (attempt / 'common_history_000/states.npz').write_bytes(b'history-proof')
        publish.write(attempt / 'initial_plan/manifest.json', {'purpose': 'pre-observation initial plan'})
        inventory = {str(path.relative_to(attempt)): publish.file_record(path)
                     for path in sorted(attempt.rglob('*')) if path.is_file()}
        publish.write(reports / 'delivery_files.json', {'schema': 'paired-delivery-sha256-v1', 'files': inventory})
        entry.update(data=str(data.relative_to(self.work)), reports=str(reports.relative_to(self.work)),
            delivery_inventory_sha256=publish.file_sha(reports / 'delivery_files.json'))
        self.entries.append(entry)
        publish.write(self.manifest, {'accepted_scenes': self.entries, 'target_scene_count': 200})
        return attempt

    def curate(self):
        return curate.curate_dataset(self.manifest, self.curation.parent, dataset_purpose='evaluation')

    def run_publish(self, **kwargs):
        return publish.publish_dataset(self.manifest, self.output, curation_manifest=self.curation, **kwargs)

    def test_clean_whole_scene_links_raw_bytes_and_curator_compatibility(self):
        attempt = self.add_scene(); self.curate()
        original = attempt / 'data/scene_0000/spec.json'
        original_sha = publish.file_sha(original)
        result = self.run_publish()
        self.assertEqual(result['published_scene_count'], 1)
        self.assertFalse(result['target_met'])
        self.assertEqual(result['remaining_scene_count'], 199)
        self.assertEqual(publish.file_sha(original), original_sha)
        self.assertEqual(original.stat().st_ino, (self.output / 'scene_0000/spec.json').stat().st_ino)
        self.assertFalse(any('attempt_' in p.name or 'candidate_' in p.name for p in self.output.rglob('*')))
        self.assertTrue((self.output / 'scene_0000/qa/history/common_history_000/states.npz').exists())
        self.assertTrue((self.output / 'scene_0000/qa/initial_plan/manifest.json').exists())
        self.assertTrue(publish.verify_published_dataset(self.output)['verified'])
        records, excluded = curate.load_scenes(self.output / 'dataset_manifest.json')
        self.assertEqual(excluded, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]['model_branches']), 5)
        self.assertNotIn('maintain_repeat', records[0]['model_branches'])
        self.assertEqual(records[0]['qa_only_branches'], ['maintain_repeat'])
        for row in publish.read(self.output / 'review.json'):
            for path in publish.review_links(row):
                self.assertTrue((self.output / path).is_file(), path)

    def test_only_explicit_curated_representatives_with_all_branches(self):
        self.add_scene(); self.add_scene('scene_0001')
        result = self.curate()
        self.assertEqual(result['retained_scene_count'], 1)
        publication = self.run_publish()
        self.assertEqual(publication['published_scene_count'], 1)
        self.assertFalse((self.output / 'scene_0001').exists())
        self.assertEqual(len(list((self.output / 'scene_0000/branches').iterdir())), 6)

    def test_requires_explicit_curation_no_silent_fitting(self):
        self.add_scene()
        with self.assertRaisesRegex(ValueError, 'explicit completed curation'):
            publish.publish_dataset(self.manifest, self.output)
        self.assertFalse(self.output.exists())

    def test_stale_curation_rejected(self):
        self.add_scene(); self.curate(); self.add_scene('scene_0001', delta=5.)
        with self.assertRaisesRegex(ValueError, 'does not bind'):
            self.run_publish()

    def test_idempotent_and_verified_update_to_two_scenes(self):
        self.add_scene(); self.curate(); first = self.run_publish()
        self.assertEqual(self.run_publish()['publication_action'], 'verified_existing')
        self.add_scene('scene_0001', delta=5.); self.curate()
        second = self.run_publish()
        self.assertEqual(second['publication_action'], 'updated')
        self.assertEqual(second['published_scene_count'], 2)
        self.assertEqual(second['source_scene_seals']['scene_0000'], first['source_scene_seals']['scene_0000'])
        self.assertFalse(list(self.base.glob('.dataset.previous-*')))
        self.assertFalse(list(self.base.glob('.dataset.publishing-*')))

    def test_raw_mutation_rejected_even_with_copy_and_identical_manifest(self):
        attempt = self.add_scene(); self.curate(); self.run_publish(link_mode='copy')
        (attempt / 'data/scene_0000/branches/maintain/states.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed sealed'):
            self.run_publish(link_mode='copy')
        self.assertTrue(publish.verify_published_dataset(self.output)['verified'])

    def test_source_symlink_rejected(self):
        attempt = self.add_scene(); self.curate()
        image = attempt / 'data/scene_0000/branches/maintain/camera/CAM_FRONT/00000.png'
        outside = self.base / 'outside.png'; outside.write_bytes(image.read_bytes())
        image.unlink(); image.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.run_publish()
        self.assertFalse(self.output.exists())

    def test_output_overlap_and_unmanaged_or_extra_content_refused(self):
        self.add_scene(); self.curate()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            publish.publish_dataset(self.manifest, self.work / 'clean', curation_manifest=self.curation)
        self.output.mkdir(); (self.output / 'unrelated.txt').write_text('do not touch')
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.run_publish()
        self.assertEqual((self.output / 'unrelated.txt').read_text(), 'do not touch')
        shutil.rmtree(self.output); self.run_publish()
        (self.output / 'new-user-file.txt').write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'extra/missing'):
            self.run_publish()

    def test_cross_filesystem_copy_fallback_and_strict_hardlink_error(self):
        attempt = self.add_scene(); self.curate()
        with patch.object(publish.os, 'link', side_effect=OSError(errno.EXDEV, 'cross device')):
            result = self.run_publish()
        self.assertGreater(result['transfer']['copied_files'], 0)
        self.assertNotEqual((attempt / 'data/scene_0000/spec.json').stat().st_ino,
                            (self.output / 'scene_0000/spec.json').stat().st_ino)
        shutil.rmtree(self.output)
        with patch.object(publish.os, 'link', side_effect=OSError(errno.EXDEV, 'cross device')):
            with self.assertRaises(OSError):
                self.run_publish(link_mode='hardlink')
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.base.glob('.dataset.publishing-*')))

    def test_failure_during_final_rename_restores_old_snapshot(self):
        self.add_scene(); self.curate(); self.run_publish()
        old = publish.file_sha(self.output / 'checksums.json')
        self.add_scene('scene_0001', delta=5.); self.curate()
        rename = Path.rename
        def fail(source, target):
            if source.name.startswith('.dataset.publishing-'):
                raise OSError('simulated final rename failure')
            return rename(source, target)
        with patch.object(Path, 'rename', fail):
            with self.assertRaisesRegex(OSError, 'simulated'):
                self.run_publish()
        self.assertEqual(publish.file_sha(self.output / 'checksums.json'), old)
        self.assertTrue(publish.verify_published_dataset(self.output)['verified'])

    def test_original_seal_change_rejected_on_update(self):
        self.add_scene(); self.curate(); self.run_publish(link_mode='copy')
        self.entries[0]['delivery_inventory_sha256'] = '0' * 64
        publish.write(self.manifest, {'accepted_scenes': self.entries, 'target_scene_count': 200})
        self.curate()
        with self.assertRaisesRegex(ValueError, 'Original delivery inventory'):
            self.run_publish()

    def test_traversal_in_sealed_inventory_rejected(self):
        attempt = self.add_scene(); self.curate()
        seal = attempt / 'reports/delivery_files.json'
        inventory = publish.read(seal); inventory['files']['../../elsewhere'] = {'size': 0, 'sha256': 'x'}
        publish.write(seal, inventory)
        self.entries[0]['delivery_inventory_sha256'] = publish.file_sha(seal)
        publish.write(self.manifest, {'accepted_scenes': self.entries, 'target_scene_count': 200}); self.curate()
        with self.assertRaisesRegex(ValueError, 'Unsafe relative path'):
            self.run_publish()

    def test_raw_change_between_source_check_and_transfer_is_rejected(self):
        self.add_scene(); self.curate()
        copy_file = publish._copy_file
        altered = []
        def mutate(source, target, mode, counters):
            if source.suffix == '.png' and not altered:
                source.write_bytes(b'changed after initial seal verification')
                altered.append(str(source))
            return copy_file(source, target, mode, counters)
        with patch.object(publish, '_copy_file', mutate):
            with self.assertRaisesRegex(ValueError, 'differ from the original seal'):
                self.run_publish()
        self.assertTrue(altered)
        self.assertFalse(self.output.exists())

    def test_preserves_declared_train_eval_splits_without_refitting(self):
        self.add_scene(); self.add_scene('scene_0001', delta=5.)
        self.curate()
        curation = publish.read(self.curation)
        curation['dataset_purpose'] = 'train_eval'
        curation['splits'] = {'train': ['scene_0000'], 'test': ['scene_0001']}
        for row in curation['retained_scenes']:
            row['split'] = 'train' if row['scene_id'] == 'scene_0000' else 'test'
        publish.write(self.curation, curation)
        result = self.run_publish()
        self.assertEqual(result['splits'], curation['splits'])
        self.assertEqual(publish.read(self.output / 'splits/test.json')['scene_ids'], ['scene_0001'])
        self.assertFalse((self.output / 'splits/evaluation.json').exists())


if __name__ == '__main__':
    unittest.main()
