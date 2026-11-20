"""Regressions for the non-mutating renderer preparation and cache-read receipt."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dfao_renderer as renderer

class RendererTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)/'carla'
        self.logs = Path(tmp.name)/'logs'
        self.logs.mkdir()
        for sub in (renderer.SHIPPING.parent, renderer.CACHE.parent, Path('CarlaUE4/Content')):
            (self.base/sub).mkdir(parents=True, exist_ok=True)
        self.source = b'0123456789'
        self.fixed = b'01replacement6789'
        (self.base/renderer.CACHE).write_bytes(self.source)
        (self.base/renderer.SHIPPING).write_bytes(b'shipping')
        p = Path(tmp.name)/'patch.json'
        self.delta = {'policy': renderer.POLICY, 'original_sha256': renderer.digest(self.source),
                      'corrected_sha256': renderer.digest(self.fixed), 'blocks': [
                {'offset':2,'original_bytes':4,'compressed_block':base64.b64encode(b'replacement').decode()}]}
        self.patch_path = p
        p.write_text(json.dumps(self.delta))
        constants = patch.multiple(renderer, STOCK=self.delta['original_sha256'],
                                   CORRECTED=self.delta['corrected_sha256'], DELTA=p)
        constants.start()
        self.addCleanup(constants.stop)

    def test_sparse_reconstruction_and_identity(self):
        self.assertEqual(renderer.correct(self.source), self.fixed)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            renderer.correct(b'other')
        self.delta['blocks'][0]['offset'] = 3
        self.patch_path.write_text(json.dumps(self.delta))
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            renderer.correct(self.source)

    def test_writes_only_private_files_and_reuses_them(self):
        runtime = renderer.prepare(self.base, self.logs)
        before = (runtime/renderer.CACHE).stat()
        self.assertTrue((runtime/'CarlaUE4/Content').is_symlink())
        self.assertEqual((runtime/renderer.CACHE).read_bytes(), self.fixed)
        self.assertEqual((self.base/renderer.CACHE).read_bytes(), self.source)
        self.assertFalse((self.base/renderer.SHIPPING).samefile(runtime/renderer.SHIPPING))
        self.assertEqual(renderer.prepare(self.base,self.logs), runtime)
        self.assertEqual((runtime/renderer.CACHE).stat().st_ino, before.st_ino)
        (runtime/renderer.CACHE).write_bytes(self.source)
        with self.assertRaisesRegex(ValueError,'validation'):
            renderer.prepare(self.base,self.logs)

    def test_event_requires_a_real_file_read_and_matching_launch(self):
        runtime = renderer.prepare(self.base,self.logs)
        watch = renderer.CacheRead(runtime/renderer.CACHE)
        self.addCleanup(watch.close)
        self.assertFalse(watch.observe())
        (self.base/renderer.CACHE).read_bytes()
        self.assertFalse(watch.observe())
        (runtime/renderer.CACHE).read_bytes()
        self.assertTrue(watch.observe())
        value = renderer.receipt(self.base, runtime)
        renderer.verify(value, str(runtime/renderer.SHIPPING))
        for change in ({'opened_and_closed_readonly': False}, {'cache_sha256': renderer.STOCK}):
            with self.assertRaises(ValueError):
                renderer.verify({**value,**change}, str(runtime/renderer.SHIPPING))
        with self.assertRaises(ValueError):
            renderer.verify(value, str(self.base/renderer.SHIPPING))

if __name__ == '__main__':
    unittest.main()
