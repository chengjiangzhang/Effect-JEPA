"""Regression cases for five effective actions, including non-reference pairs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

if __package__:
    from .action_separation import CONTRACT, audit_separation
    from .audit_realized_actions import audit_scene
else:
    from action_separation import CONTRACT, audit_separation
    from audit_realized_actions import audit_scene


NAMES = ['maintain', 'brake', 'accelerate', 'left', 'right']


def fixture():
    times = np.arange(121)*.05-2.
    future = np.maximum(times, 0.)
    states = {}
    for name, acceleration in zip(NAMES, [0., -1.6, 1.6, -.8, .8]):
        state = {key: np.zeros((121,1,3)) for key in ('position','velocity','rotation_deg','control')}
        state['time_s'] = times.copy()
        motion_time = np.minimum(future, 5./-acceleration) if acceleration < 0 else future
        state['position'][:,0,0] = 5.*np.minimum(times,0.) + 5.*motion_time + .5*acceleration*motion_time**2
        state['velocity'][:,0,0] = np.maximum(0.,5.+acceleration*future)
        state['control'][40:-1,0,0 if acceleration >= 0 else 2] = abs(acceleration)*.4
        states[name] = state
    states['maintain_repeat'] = copy.deepcopy(states['maintain'])
    metadata = {name: {'collisions':[]} for name in states}
    return states, metadata


class SeparationTests(unittest.TestCase):
    def audit(self, states, metadata):
        return audit_separation(states, NAMES, metadata, copy.deepcopy(CONTRACT))

    def test_all_ten_pairs_pass_and_repeat_is_not_a_sixth_action(self):
        result = self.audit(*fixture())
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['passed_pair_count'],10)
        self.assertFalse(any('maintain_repeat' in p['branches'] for p in result['pairs']))

    def test_nonreference_duplicate_fails_whole_scene(self):
        states, meta = fixture(); states['right'] = copy.deepcopy(states['left'])
        result = self.audit(states, meta)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['passed_pair_count'],9)
        self.assertEqual([p['branches'] for p in result['pairs'] if not p['passed']],[['left','right']])

    def test_different_controls_with_centimetre_motion_do_not_pass(self):
        states, meta = fixture()
        states['right']['position'] = states['left']['position'].copy() + .01
        states['right']['velocity'] = states['left']['velocity'].copy()
        self.assertEqual(self.audit(states,meta)['status'],'failed')

    def test_motion_without_control_difference_does_not_pass(self):
        states, meta = fixture(); states['right']['control'] = states['left']['control'].copy()
        pair = next(p for p in self.audit(states,meta)['pairs'] if p['branches']==['left','right'])
        self.assertFalse(pair['passed'])
        self.assertEqual(pair['failure_reasons'],['insufficient_sustained_executed_control_difference'])

    def test_last_frame_or_short_spike_is_insufficient(self):
        states, meta = fixture(); states['right'] = copy.deepcopy(states['left'])
        states['right']['position'][-1,0,0] += 20.
        states['right']['control'][-1,0,0] = 1.
        self.assertEqual(self.audit(states,meta)['passed_pair_count'],9)

    def test_contact_cannot_supply_the_required_motion_separation(self):
        states, meta = fixture(); meta['right']['collisions'] = [{'time_s':.3,'role':'ego','other_role':'rear'}]
        result = self.audit(states,meta)
        self.assertEqual(result['status'],'failed')
        self.assertTrue(all(not p['passed'] for p in result['pairs'] if 'right' in p['branches']))

    def test_late_contact_keeps_already_distinct_actions(self):
        states, meta = fixture(); meta['right']['collisions'] = [{'time_s':3.8,'role':'ego','other_role':'rear'}]
        self.assertEqual(self.audit(states,meta)['status'],'passed')

    def test_repeat_noise_raises_evidence_requirement(self):
        states, meta = fixture(); states['maintain_repeat']['position'][41:,0,0] += .3
        result = self.audit(states,meta)
        self.assertAlmostEqual(result['effective_thresholds']['position_m'],1.5)

    def test_missing_evidence_or_modified_thresholds_fail_closed(self):
        states, meta = fixture(); meta['right'].pop('collisions')
        with self.assertRaises(ValueError): self.audit(states,meta)
        states, meta = fixture(); contract = copy.deepcopy(CONTRACT); contract['position_threshold_m']=.01
        with self.assertRaises(ValueError): audit_separation(states,NAMES,meta,contract)
        states['right']['time_s'][-1]=4.5
        with self.assertRaises(ValueError): self.audit(states,meta)

    def test_scene_audit_enforces_new_contract_but_preserves_legacy_reports(self):
        states, metadata = fixture(); states['right']=copy.deepcopy(states['left'])
        cfg = {'vehicle_roles':['ego'],'vehicles':[{'role':'ego'}], 'action':{},
               'spec_sha256':'fixture','code_sha256':{'fixture':'source'}}
        with tempfile.TemporaryDirectory() as directory:
            scene=Path(directory)/'scene_0000'; scene.mkdir()
            path=scene/'spec.json'; path.write_text(json.dumps(cfg))
            for name,state in states.items():
                branch=scene/'branches'/name;branch.mkdir(parents=True)
                np.savez_compressed(branch/'states.npz',**state)
                meta={**metadata[name], 'status':'complete','branch':name,'vehicle_roles':['ego'],
                      'spec_sha256':'fixture','code_sha256':cfg['code_sha256']}
                (branch/'meta.json').write_text(json.dumps(meta))
            self.assertEqual(audit_scene(scene)['status'],'passed')
            cfg['action_separation_contract']=copy.deepcopy(CONTRACT);path.write_text(json.dumps(cfg))
            result=audit_scene(scene)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['action_separation']['passed_pair_count'],9)
            self.assertEqual(result['model_branches'],NAMES)


if __name__ == '__main__':
    unittest.main(verbosity=2)
