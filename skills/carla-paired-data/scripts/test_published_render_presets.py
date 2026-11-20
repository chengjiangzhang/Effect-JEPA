import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from runtime import collect_multicam as cm


class Selected(Exception):
    pass


class PublishedRenderPresetTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.sid='scene_0006'
        (self.root/self.sid).mkdir()
        self.default=self.root/'render_preset.json'
        self.override=self.root/self.sid/'render_preset.json'
        self.default.write_bytes(b'old');self.override.write_bytes(b'new')
        self.digest=hashlib.sha256(b'new').hexdigest()
        self.spec={'scene_id':self.sid,'render_preset':{'file':'render_preset.json','sha256':self.digest}}
        self.manifest={'status':'published_snapshot','scene_render_presets':{self.sid:{'file':self.sid+'/render_preset.json','sha256':self.digest}}}

    def validate(self):
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        return cm.check_frozen_render_preset(self.root,self.spec)

    def selected(self,path):
        with patch.object(cm,'load_render_preset',side_effect=Selected) as loader:
            with self.assertRaises(Selected):self.validate()
            loader.assert_called_once_with(path)

    def test_binding(self):
        self.selected(self.override)

    def test_legacy_default(self):
        self.spec['render_preset']['sha256']=hashlib.sha256(b'old').hexdigest()
        self.selected(self.default)

    def test_cannot_override_native_collection(self):
        self.manifest['status']='planned'
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_path_or_sha(self):
        for field,value in [('file','../render_preset.json'),('sha256','0'*64)]:
            with self.subTest(field=field):
                binding=self.manifest['scene_render_presets'][self.sid];prior=binding[field];binding[field]=value
                with self.assertRaises(ValueError):self.validate()
                binding[field]=prior

    def test_corrupt_preset(self):
        self.override.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.validate()
