"""Offline process-budget and progress checks; no simulator is connected."""
import io
import json
import subprocess
import unittest
from unittest.mock import patch

if __package__:
    from . import collect_multicam as capture
    from . import collect_multicam_batch as batch
else:
    import collect_multicam as capture
    import collect_multicam_batch as batch


class CollectionLivenessTests(unittest.TestCase):
    def test_child_has_bounded_budget_above_observed_slow_capture(self):
        command = ['python', 'collect_multicam.py', '--scene', 'scene_0003']
        stream = io.StringIO()
        with patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess(command, 0)) as run:
            self.assertEqual(batch.run_branch_process(command, stream), (0, False))
        run.assert_called_once_with(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=600, check=False, start_new_session=True)

    def test_timeout_is_distinguished_from_child_error(self):
        command = ['python', 'collect_multicam.py']
        with patch.object(batch.subprocess, 'run', side_effect=subprocess.TimeoutExpired(command, 600)):
            self.assertEqual(batch.run_branch_process(command, io.StringIO()), (-1, True))
        with patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess(command, 1)):
            self.assertEqual(batch.run_branch_process(command, io.StringIO()), (1, False))

    def test_os_launch_error_is_not_falsely_labelled_timeout(self):
        with patch.object(batch.subprocess, 'run', side_effect=FileNotFoundError('missing executable')):
            with self.assertRaises(FileNotFoundError):
                batch.run_branch_process(['missing-python'], io.StringIO())

    def test_progress_is_flushed_and_does_not_claim_queued_images_saved(self):
        with patch.object(capture.time, 'monotonic', return_value=500.), patch('builtins.print') as output:
            capture.log_capture_progress('scene_0003', 'maintain', 'sample_queued', 200.,
                                         sample_index=11, sample_count=13, physics_step=110,
                                         physics_frame_id=900, relative_time_s=3.5)
        output.assert_called_once()
        self.assertTrue(output.call_args.kwargs['flush'])
        row=json.loads(output.call_args.args[0])
        self.assertEqual(row['elapsed_wall_s'],300.)
        self.assertEqual(row['phase'],'sample_queued')
        self.assertEqual(row['sample_index'],11)
        self.assertEqual(row['relative_time_s'],3.5)
        self.assertNotIn('images_saved',row)
        self.assertEqual(row['event'],'capture_progress')


if __name__ == '__main__':
    unittest.main()
