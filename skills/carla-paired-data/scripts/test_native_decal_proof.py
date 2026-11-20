import copy
import json
from pathlib import Path
import tempfile
import unittest

import native_decal_proof as proof


def header():
    return {'kind':'decal_order_start','policy':proof.POLICY,'target':'0x3bf64f0','entry_verified':True,
            'preserved_primary_order':proof.PRIMARY,'stable_ties':proof.TIES}


def draw():
    return {'kind':'decal_order_apply','ordinal':1,'actor':'SceneCaptureCamera','actor_number':23,
            'frame':100,'time':5.0,'stage':0,'count':8,'moved':7,'ordered_input_hash':'0123456789abcdef'}


def lines(rows):
    return ''.join(json.dumps(row)+'\n' for row in rows)


class DecalProofTests(unittest.TestCase):
    def test_no_multidecal_list_is_recorded_as_zero_not_an_effect(self):
        result=proof.trace_audit(lines([header()]),header())
        self.assertEqual(result['calls'],0)
        self.assertEqual(result['camera_lists'],0)

    def test_valid_apply_and_optional_matching_shutdown(self):
        rows=[header(),draw()]
        self.assertEqual(proof.trace_audit(lines(rows),header())['camera_lists'],1)
        rows.append({'kind':'decal_order_end','calls':1,'changed_lists':1})
        self.assertTrue(proof.trace_audit(lines(rows),header())['footer_present'])

    def test_native_error_incomplete_sequence_and_changed_policy_fail(self):
        for rows in ([header(),{'kind':'decal_order_error','source_line':123}],
                     [header(),{**draw(),'ordinal':2}], [header(),{**draw(),'moved':9}],
                     [header(),{**draw(),'ordered_input_hash':'unknown'}],
                     [header(),{**draw(),'time':float('nan')}],
                     [header(),draw(),{'kind':'decal_order_end','calls':0,'changed_lists':0}],
                     [header(),header()]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):proof.trace_audit(lines(rows),header())
        with self.assertRaises(ValueError):proof.trace_audit(lines([header()])[:-1],header())
        with self.assertRaises(ValueError):proof.validate_header({**header(),'policy':'old_or_wrong'})

    def test_owned_process_snapshot_and_post_snapshot_shutdown(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder=root/'weather_rng';folder.mkdir();path=folder/'decal-order-123.jsonl'
            path.write_text(lines([header(),draw()]))
            launch=proof.launch_evidence(root,123);value=proof.snapshot(launch,123)
            path.write_text(lines([header(),draw(),{'kind':'decal_order_end','calls':1,'changed_lists':1}]))
            proof.verify(value,launch,123)
            with self.assertRaises(ValueError):proof.verify(value,launch,456)
            changed=copy.deepcopy(value);changed['trace_snapshot']=changed['trace_snapshot'].replace('0123456789abcdef','fedcba9876543210')
            with self.assertRaises(ValueError):proof.verify(changed,launch,123)


if __name__=='__main__':unittest.main()
