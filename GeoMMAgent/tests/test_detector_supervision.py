"""Exited workers recover within a budget;live and occupied services stay alone."""
from pathlib import Path
import sys,unittest
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from detector_supervision import DetectorSupervisor


class DetectorSupervisionTests(unittest.TestCase):
    def test_live_child_and_foreign_occupied_port_never_start_another_worker(self):
        child=Mock();child.poll.return_value=None;restart=Mock();port=Mock(return_value=True)
        supervisor=DetectorSupervisor(lambda:child,restart,port)
        supervisor.tick();restart.assert_not_called();port.assert_not_called()
        child.poll.return_value=1;supervisor.tick();restart.assert_not_called()
        self.assertEqual(supervisor.state,'port_occupied')

    def test_exited_owned_child_is_replaced_then_monitored(self):
        child=Mock();child.poll.return_value=1;holder=[child];new=Mock();new.poll.return_value=None
        restart=Mock(side_effect=lambda:holder.__setitem__(0,new))
        supervisor=DetectorSupervisor(lambda:holder[0],restart,lambda:False)
        supervisor.tick();supervisor.tick()
        restart.assert_called_once();self.assertEqual(supervisor.state,'child_running')

    def test_failed_restarts_are_bounded_and_can_resume_after_window(self):
        child=Mock();child.poll.return_value=1;clock=[0];restart=Mock(side_effect=RuntimeError('private diagnostic'))
        supervisor=DetectorSupervisor(lambda:child,restart,lambda:False,clock=lambda:clock[0])
        for _ in range(8):supervisor.tick()
        self.assertEqual(restart.call_count,3);self.assertEqual(supervisor.state,'retry_budget_wait')
        clock[0]=301;supervisor.tick();self.assertEqual(restart.call_count,4)


if __name__=='__main__':unittest.main()
