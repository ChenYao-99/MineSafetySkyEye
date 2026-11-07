"""Source-only selection and optimizer continuation guards."""
from pathlib import Path
import sys,unittest,copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import training_sufficiency as exp


class SufficiencyTests(unittest.TestCase):
    def test_checkpoint_selection_ignores_any_target_score(self):
        rows=[{'update':1200,'source_selection':{'macro_f1':.8},'target_f1':.1},
              {'update':4800,'source_selection':{'macro_f1':.7},'target_f1':1.}]
        self.assertEqual(exp.choose_source_checkpoint(rows)['update'],1200)

    def test_checkpoint_ties_choose_earliest(self):
        rows=[{'update':4800,'source_selection':{'macro_f1':.8}},
              {'update':1200,'source_selection':{'macro_f1':.8}}]
        self.assertEqual(exp.choose_source_checkpoint(rows)['update'],1200)

    def test_calibration_and_selection_parents_are_disjoint(self):
        for g,n in enumerate([256,794,256,256,256,256]):
            a,b=exp.split_helper.meta_split(n,g)
            self.assertFalse(set(a)&set(b));self.assertEqual(set(a)|set(b),set(range(n)))

    def test_training_reference_loss_retains_fractional_labels(self):
        labels=np.full((2,64,64),.25,np.float32)
        self.assertLess(exp.soft_loss(np.full_like(labels,.25),labels),exp.soft_loss(np.full_like(labels,.01),labels))

    def test_restart_retains_adam_moments_and_sets_live_lr(self):
        import torch
        x=torch.nn.Parameter(torch.tensor([1.]));o=torch.optim.AdamW([x],lr=.001)
        for _ in range(4):o.zero_grad();x.square().sum().backward();o.step()
        state=copy.deepcopy(o.state[x]);exp.start_continuation(o,4,.0003)
        self.assertAlmostEqual(o.param_groups[0]['lr'],.0003)
        self.assertTrue(torch.equal(o.state[x]['exp_avg'],state['exp_avg']))
        self.assertTrue(torch.equal(o.state[x]['step'],state['step']))

    def test_resume_keeps_continuation_numerics(self):
        import torch
        def setup():
            a=torch.nn.Parameter(torch.tensor([1.]));o=torch.optim.AdamW([a],lr=.001);s=exp.start_continuation(o,8,.0003)
            return a,o,s
        a,o,s=setup()
        for _ in range(4):o.zero_grad();a.square().sum().backward();o.step();s.step()
        cp=(a.detach().clone(),copy.deepcopy(o.state_dict()),copy.deepcopy(s.state_dict()))
        for _ in range(4):o.zero_grad();a.square().sum().backward();o.step();s.step()
        b,bo,bs=setup();b.data.copy_(cp[0]);bo.load_state_dict(cp[1]);bs.load_state_dict(cp[2])
        for _ in range(4):bo.zero_grad();b.square().sum().backward();bo.step();bs.step()
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(o.param_groups[0]['lr'],bo.param_groups[0]['lr'])

    def test_resume_payload_loads_in_weights_only_mode(self):
        import torch,io
        payload={'rng':np.random.default_rng(42).bit_generator.state,'schedule':bytes(range(64)),
                 'visits':[4,4,0,4,4,4],'torch_rng':torch.get_rng_state(),'curve':[{'update':1200,'source_selection':{'macro_f1':.7}}]}
        stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
        restored=torch.load(stream,map_location='cpu',weights_only=True)
        self.assertEqual(restored['rng'],payload['rng']);self.assertEqual(restored['schedule'],payload['schedule'])


if __name__=='__main__':unittest.main()
