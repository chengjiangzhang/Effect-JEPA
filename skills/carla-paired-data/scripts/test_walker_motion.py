import unittest
import numpy as np
from runtime.walker_motion import examine


class WalkerMotionTests(unittest.TestCase):
    def trace(self, speed=1., walk=0., t0=0.):
        times=np.arange(31)*.05+t0
        c=np.zeros((31,1,5)); c[:-1,0,0]=1.; c[:-1,0,3]=speed
        observed=np.zeros_like(c); observed[1:]=c[:-1]
        p=np.zeros((31,1,3)); p[:,0,0]=walk*(times-t0)
        return dict(time_s=times, position=p, control=c, observed_control=observed)

    def test_true_wait_and_motion(self):
        for speed,walk in [(0.,0.),(1.2,1.15)]:
            self.assertTrue(examine(self.trace(speed,walk),0,'walker')['passed'])

    def test_acknowledged_stall(self):
        r=examine(self.trace(walk=.001),0,'walker')
        self.assertFalse(r['passed'])
        self.assertEqual(r['sustained_stalls'][0]['end_s'],1.5)

    def test_role_specific_contact_only(self):
        data=self.trace()
        event=dict(time_s=.2,role='ego',other_role='walker')
        self.assertTrue(examine(data,0,'walker',[event],['ego'])['passed'])
        for change in [dict(other_role='other_walker'),dict(role='static.fence'),dict(time_s=1.2)]:
            self.assertFalse(examine(data,0,'walker',[event|change],['ego'])['passed'])

    def test_reversal_and_final_sentinel(self):
        d=self.trace()
        d['control'][::2,0,0]=-1
        d['control'][-1]=0
        d['observed_control'][1:]=d['control'][:-1]
        self.assertTrue(examine(d,0,'walker')['passed'])

    def test_missing_acknowledgement(self):
        d=self.trace();d['observed_control'][:]=0
        self.assertFalse(examine(d,0,'walker')['passed'])

    def test_brief_startup(self):
        d=self.trace(walk=1.);d['position'][:6]=0.
        self.assertTrue(examine(d,0,'walker')['passed'])
