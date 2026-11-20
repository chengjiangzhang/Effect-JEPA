"""Offline driving-prefix timing tests with a fake world; no simulator RPC."""
from types import SimpleNamespace
from unittest import mock
import unittest

import numpy as np

if __package__:
    from . import collect_multicam as cm
    from . import preflight_physics as preflight
else:
    import collect_multicam as cm
    import preflight_physics as preflight


class PrefixTimingTests(unittest.TestCase):
    def test_history_only_observes_t0_without_evaluating_or_executing_future(self):
        frame = [0]
        calls, applied, auxiliary_times = [], [], []
        def tick(timeout):
            frame[0] += 1
            return frame[0]
        world = SimpleNamespace(tick=tick, get_snapshot=lambda: SimpleNamespace(
            frame=100+frame[0], timestamp=SimpleNamespace(elapsed_seconds=10+frame[0]*.05)))
        def observe(actors):
            return {**{k: np.full((1, 3), float(frame[0])) for k in
                       ('position', 'rotation_deg', 'velocity', 'angular_velocity', 'acceleration')},
                    'transform_world_vehicle': np.eye(4)[None]}
        def controls(obs, branch, time_s, extents, ego_base_speed):
            self.assertLess(time_s, 0.)
            self.assertEqual(branch, 'maintain')
            calls.append(time_s)
            return np.ones((1, 3)), [7.], [{'phase': 'common_history'}]
        auxiliary = SimpleNamespace(inject_observation=lambda obs, t: obs,
            record=lambda t, final: auxiliary_times.append((t, final)) or np.zeros((0, 5)),
            advance_commands=lambda values, t: None)
        actuation = {**{key: np.zeros(1) for key in ('observed_gear', 'observed_manual_gear_shift',
                     'requested_gear', 'requested_manual_gear_shift')}, 'observed_control': np.zeros((1, 3))}
        with mock.patch.object(cm, 'observe', side_effect=observe), \
             mock.patch.object(cm, 'observe_actuation', return_value=actuation), \
             mock.patch.object(cm, 'apply', side_effect=lambda *args: applied.append(frame[0])):
            arrays, diagnostics, speed = preflight.record_physics(None, world, [object()],
                {'vehicle_roles': ['ego'], 'vehicles': [{'initial_speed_mps': 7.}]},
                'common_history', SimpleNamespace(controls=controls), np.ones((1,3)), auxiliary,
                history_only=True)
        self.assertEqual(frame[0], 40)
        self.assertEqual(len(calls), 40)
        self.assertEqual(len(applied), 40)
        self.assertEqual(arrays['position'].shape, (41, 1, 3))
        self.assertEqual(arrays['time_s'][-1], 0.)
        np.testing.assert_array_equal(arrays['position'][40], np.full((1,3),40.))
        np.testing.assert_array_equal(arrays['control'][40], 0.)
        self.assertAlmostEqual(speed, np.hypot(40., 40.))
        self.assertFalse(diagnostics[-1][0]['future_control_evaluated'])
        self.assertEqual(auxiliary_times[-1], (0., True))
        self.assertFalse(any(t > 0 for t, _ in auxiliary_times))

    def test_tick_aligned_range_and_declared_timing_agreement(self):
        for value in (.5,.75,1.,2.,3.):
            self.assertEqual(cm.motion_warmup_seconds({'driving_warmup_s':value}),value)
        for value in (0,.49,3.05,.76,True,float('nan')):
            with self.assertRaises(ValueError):cm.motion_warmup_seconds({'driving_warmup_s':value})
        with self.assertRaises(ValueError):
            cm.motion_warmup_seconds({'driving_warmup_s':2.,'timing':{'motion_warmup_s':1.}})

    def test_variable_warmup_reaches_identical_saved_history_start(self):
        for duration in (.5,.75,3.):
            with self.subTest(duration=duration):
                cfg={'driving_warmup_s':duration,'timing':{'history_s':2.,'motion_warmup_s':duration},
                     'vehicles':[{'initial_speed_mps':8.}]}
                frame=[0]
                def tick(timeout):frame[0]+=1;return frame[0]
                world=SimpleNamespace(tick=tick,get_snapshot=lambda:SimpleNamespace(frame=frame[0]))
                vector=SimpleNamespace(x=1.,y=0.,z=0.)
                bbox=SimpleNamespace(extent=vector,location=vector,
                                     rotation=SimpleNamespace(roll=0.,pitch=0.,yaw=0.))
                actor=SimpleNamespace(id=1,bounding_box=bbox,
                    get_transform=lambda:SimpleNamespace(get_forward_vector=lambda:vector))
                times=[];signal_times=[];commands=[];walker_times=[]
                controller=SimpleNamespace(controls=lambda obs,branch,t,ext:(times.append(t) or np.zeros((1,3)),None,None))
                auxiliary=SimpleNamespace(actors=[],hold_pedestrians=lambda reset:None,
                    set_signal_time=signal_times.append,inject_observation=lambda obs,t:obs,
                    warmup_commands=lambda t:walker_times.append(t) or np.zeros((0,5)),
                    advance_commands=lambda controls,t:commands.append(t))
                with mock.patch.object(cm,'apply'),mock.patch.object(cm,'apply_commands'),\
                     mock.patch.object(cm,'observe',return_value={}),\
                     mock.patch.object(preflight,'ScenarioController',return_value=controller):
                    *_,trace=preflight.initialize_motion(None,world,[actor],cfg,auxiliary)
                self.assertEqual(frame[0],21+round(duration/.05))
                self.assertEqual(len(times),round(duration/.05))
                self.assertEqual(walker_times,times)
                self.assertAlmostEqual(times[0],-2-duration)
                self.assertAlmostEqual(times[-1],-2.05)
                self.assertAlmostEqual(commands[-1],-2.)
                self.assertEqual(signal_times,[-2-duration])
                self.assertEqual(trace['motion_warmup_s'],duration)


if __name__=='__main__':unittest.main(verbosity=2)
