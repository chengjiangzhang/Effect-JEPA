import copy
import unittest
import numpy as np

import npc_merge as merge


def fixture():
    source=np.c_[np.arange(0.,201.,1.),np.full(201,3.5),np.zeros(201),np.zeros(201)]
    target=source.copy();target[:,1]=0.
    m={'policy_version':merge.VERSION,'source_route':source.tolist(),'target_route':target.tolist(),
       'valid_source_interval_s_m':[0.,200.],'earliest_start_time_s':0.,'duration_s':2.5,
       'abort_duration_s':1.8,'clear_confirmation_s':.2,'minimum_gap_m':2.,
       'front_headway_s':.6,'rear_headway_s':.6,'prediction_horizon_s':2.5,
       'comfortable_brake_m_s2':3.,'abort_max_progress':.4,'lane_half_width_m':1.75}
    config={'vehicle_roles':['ego','merger'], 'vehicles':[{'role':'ego','route':target.tolist()},
        {'role':'merger','route':source.tolist(),'route_maneuver':m}], 'driving_warmup_s':1.}
    obs={'position':np.array([[10.,0.,0.],[50.,3.5,0.]]),'velocity':np.array([[5.,0.,0.],[5.,0.,0.]]),
         'rotation_deg':np.zeros((2,3)),'bbox_offset':np.zeros((2,3)),'bbox_rotation_deg':np.zeros((2,3))}
    extents=np.array([[2.4,1.,.8],[2.4,1.,.8]])
    return config,obs,extents


class MergeTests(unittest.TestCase):
    def test_rear_approach_blocks_before_lateral_movement(self):
        cfg,obs,ext=fixture();obs['position'][0,0]=40;obs['velocity'][0,0]=10;mem={}
        for i in range(30):
            route,d=merge.decide(1,obs,cfg,ext,mem,i*.05,.05)
        self.assertEqual(mem['merger']['mode'],'waiting')
        self.assertEqual(d['reason'],'target_gap_blocked')
        self.assertEqual(d['target_gap']['checks'][0]['side'],'rear')
        np.testing.assert_array_equal(route,np.asarray(cfg['vehicles'][1]['route']))

    def test_live_deceleration_releases_gap_after_confirmation(self):
        cfg,obs,ext=fixture();obs['position'][0,0]=40;obs['velocity'][0,0]=10;mem={}
        merge.decide(1,obs,cfg,ext,mem,0.,.05)
        obs['velocity'][0,0]=0.
        for i in range(1,7):
            _,d=merge.decide(1,obs,cfg,ext,mem,i*.05,.05)
        self.assertEqual(mem['merger']['mode'],'merging')
        self.assertTrue(d['target_gap']['clear'])

    def test_risk_closing_early_can_cancel_and_return(self):
        cfg,obs,ext=fixture();mem={}
        for i in range(9): merge.decide(1,obs,cfg,ext,mem,i*.05,.05)
        obs['position'][1,1]=3.3;obs['position'][0,0]=41;obs['velocity'][0,0]=11
        _,d=merge.decide(1,obs,cfg,ext,mem,.45,.05)
        self.assertEqual(d['reason'],'cancel_early_merge_target_gap_closed')
        self.assertEqual(mem['merger']['mode'],'aborting')
        for i in range(10,35):
            obs['position'][1,1]=3.5
            merge.decide(1,obs,cfg,ext,mem,i*.05,.05)
        self.assertEqual(mem['merger']['mode'],'waiting')

    def test_committed_car_does_not_turn_back_across_traffic(self):
        cfg,obs,ext=fixture();obs['position'][1,1]=1.;obs['position'][0,0]=41;obs['velocity'][0,0]=11
        mem={'merger':{**merge.initial_state(),'mode':'merging','blend':.7}}
        _,d=merge.decide(1,obs,cfg,ext,mem,1.,.05)
        self.assertEqual(d['reason'],'continue_committed_merge')
        self.assertGreater(mem['merger']['blend'],.7)

    def test_side_by_side_body_blocks(self):
        cfg,obs,ext=fixture();obs['position'][0,0]=50
        _,d=merge.decide(1,obs,cfg,ext,{},0.,.05)
        self.assertFalse(d['target_gap']['clear'])

    def test_completed_merge_uses_receiving_road_beyond_alignment_padding(self):
        cfg,obs,ext=fixture();m=cfg['vehicles'][1]['route_maneuver']
        full=np.asarray(m['target_route']);m['target_continuation_route']=full.tolist()
        padded=full.copy();padded[140:,1]=3.5;m['target_route']=padded.tolist()
        m['valid_source_interval_s_m']=[10.,139.];obs['position'][1]=[150.,0.,0.]
        mem={'merger':{**merge.initial_state(),'mode':'complete','blend':1.}}
        route,d=merge.decide(1,obs,cfg,ext,mem,3.,.05)
        np.testing.assert_array_equal(route,full)
        self.assertEqual(d['observed_lateral_progress'],1.)

    def test_return_gap_is_rechecked_after_cancellation(self):
        cfg,obs,ext=fixture();obs['position'][0]=[49.,3.5,0.]
        mem={'merger':{**merge.initial_state(),'mode':'aborting','blend':.2}}
        _,d=merge.decide(1,obs,cfg,ext,mem,1.,.05)
        self.assertFalse(d['return_gap']['clear'])
        self.assertEqual(d['reason'],'return_blocked_continue_into_clear_target')
        self.assertEqual(mem['merger']['mode'],'merging')

    def test_insufficient_mapped_distance_does_not_start_merge(self):
        cfg,obs,ext=fixture();obs['position'][1,0]=195.;mem={}
        for t in np.arange(8)*.05:_,d=merge.decide(1,obs,cfg,ext,mem,float(t),.05)
        self.assertFalse(d['eligible']);self.assertEqual(mem['merger']['mode'],'waiting')

    def test_ego_label_not_an_input(self):
        cfg,obs,ext=fixture();mem={};_,left=merge.decide(1,obs,cfg,ext,mem,0.,.05)
        cfg['vehicle_roles'][0]='some_vehicle';cfg['vehicles'][0]['role']='some_vehicle'
        other={};_,right=merge.decide(1,obs,cfg,ext,other,0.,.05)
        self.assertEqual(mem,other)
        self.assertEqual(left['state_after'],right['state_after'])

    def test_warmup_memory_recomputed_and_tampering_rejected(self):
        cfg,obs,ext=fixture();cfg['vehicles'][1]['route_maneuver']['earliest_start_time_s']=-3.;mem={};rows=[]
        for t in -3.+np.arange(20)*.05:
            _,d=merge.decide(1,obs,cfg,ext,mem,float(t),.05)
            rows.append({'time_s':float(t),'observed':{k:v.tolist() for k,v in obs.items()},
                         'bbox_extent':ext.tolist(),'decisions':{'merger':d}})
        evidence={'schema':'npc-merge-memory-v1','warmup':rows,'history_start_state':copy.deepcopy(mem)}
        self.assertEqual(merge.replay_warmup(cfg,evidence),mem)
        evidence['warmup'][3]['decisions']['merger']['state_after']['blend']+=.01
        with self.assertRaises(ValueError):merge.replay_warmup(cfg,evidence)


class MergeReturnCoverageTests(unittest.TestCase):
    def evidence(self, end_distance=.02, cancel_time=.2, return_end=.6):
        times=np.round(np.arange(-.5,1.001,.05),8)
        distance=np.interp(times,[-.5,-.4,.2,return_end,1.],[0.,0.,.3,end_distance,end_distance])
        positions=np.c_[50.+times*5.,distance,np.zeros(len(times))]
        route=np.c_[np.arange(0.,101.),np.zeros((101,3))]
        decisions=[]
        for t in times:
            before='waiting' if t<=-.4 else 'merging' if t<=cancel_time else 'aborting'
            after='waiting' if t<-.4 else 'merging' if t<cancel_time else 'aborting'
            decisions.append({'state_before':{'mode':before},'state_after':{'mode':after}})
        return times,positions,route,decisions

    def events(self,evidence,contact=4.05):
        from interaction_coverage import measured_merge_returns
        return measured_merge_returns(*evidence,contact_s=contact)

    def test_waiting_recentring_is_not_a_return(self):
        evidence=list(self.evidence());times,positions,_,decisions=evidence
        positions[:,1]=np.linspace(.039006,.008615,len(times))
        for d in decisions:d['state_before']['mode']=d['state_after']['mode']='waiting'
        self.assertEqual(self.events(evidence),[])

    def test_real_return_uses_applied_previous_command(self):
        events=self.events(self.evidence())
        self.assertEqual(len(events),1)
        self.assertAlmostEqual(events[0]['start_s'],.2)
        self.assertAlmostEqual(events[0]['end_s'],.6)
        self.assertAlmostEqual(events[0]['return_distance_m'],.28)
        self.assertTrue(events[0]['returned_to_source_centre'])

    def test_partial_return_does_not_claim_centre_completion(self):
        event=self.events(self.evidence(end_distance=.12))[0]
        self.assertFalse(event['returned_to_source_centre'])
        self.assertAlmostEqual(event['return_distance_m'],.18)

    def test_later_abort_cannot_explain_earlier_motion(self):
        self.assertEqual(self.events(self.evidence(cancel_time=.6)),[])

    def test_cancel_command_without_motion_is_not_return(self):
        evidence=list(self.evidence());evidence[1][:,1]=0.
        self.assertEqual(self.events(evidence),[])

    def test_small_physical_noise_under_abort_is_not_return(self):
        evidence=list(self.evidence());evidence[1][:,1]*=.1
        self.assertEqual(self.events(evidence),[])

    def test_short_motion_is_not_sustained_return(self):
        self.assertEqual(self.events(self.evidence(return_end=.4)),[])

    def test_pause_splits_continuous_return_descriptor(self):
        evidence=list(self.evidence(return_end=.55,end_distance=.23))
        times,positions,_,_=evidence
        index=int(np.flatnonzero(times==.4)[0])
        positions[index,1]=positions[index-1,1]
        self.assertEqual(self.events(evidence),[])

    def test_missing_recorded_start_does_not_assert_outward_excursion(self):
        evidence=list(self.evidence())
        for d in evidence[3]:
            for key in ('state_before','state_after'):
                if d[key]['mode']=='waiting':d[key]['mode']='merging'
        self.assertEqual(self.events(evidence),[])

    def test_postcontact_motion_is_excluded(self):
        self.assertEqual(self.events(self.evidence(),contact=.4),[])

    def test_pre_t0_motion_is_excluded(self):
        evidence=list(self.evidence());evidence[0]-=1.
        self.assertEqual(self.events(evidence),[])

    def test_misaligned_evidence_rejected(self):
        evidence=list(self.evidence());evidence[3]=evidence[3][:-1]
        with self.assertRaises(ValueError):self.events(evidence)


if __name__=='__main__':unittest.main()
