"""Meaningful role/chronology checks for the qualification gate."""
import copy
import unittest
from phase1_audit import requested_use, same_instant, bounds_intersect


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.core = dict(role='development', reserved=False, integrity_passed=True,
                         paired_input_passed=True, reference_kind='published_partial_accident_core',
                         positive_pixels=38, chronology_passed=True, existing_scope_admitted=True,
                         verified_accident_negative=False, full_boundary_verified=False)

    def test_fractional_timestamp_padding_is_not_a_different_acquisition(self):
        self.assertTrue(same_instant('2020-11-14T08:32:21.024Z', '2020-11-14T08:32:21.024000Z'))
        self.assertFalse(same_instant('2020-11-14T08:32:21.024Z', '2020-11-15T08:32:21.024000Z'))
        self.assertFalse(same_instant('', ''))

    def test_core_does_not_imply_complete_boundary_or_negative(self):
        self.assertTrue(requested_use(self.core, 'accident_positive_core')[0])
        self.assertFalse(requested_use(self.core, 'whole_accident_evaluation')[0])
        self.assertFalse(requested_use(self.core, 'accident_negative')[0])

    def test_accident_must_fall_between_images(self):
        self.core['chronology_passed'] = False
        self.assertFalse(requested_use(self.core, 'accident_positive_core')[0])

    def test_heldout_role_wins_over_admission(self):
        self.core['role'] = 'reserved_final_test'
        self.assertFalse(requested_use(self.core, 'accident_positive_core')[0])

    def test_expected_work_is_not_safe_background(self):
        work = copy.deepcopy(self.core)
        work.update(reference_kind='government_mining_clearing_weak_context', context_pixels=420)
        self.assertTrue(requested_use(work, 'work_context')[0])
        self.assertFalse(requested_use(work, 'accident_negative')[0])

    def test_boolean_flag_cannot_promote_reference_semantics(self):
        self.core['full_boundary_verified'] = True
        self.assertFalse(requested_use(self.core, 'whole_accident_evaluation')[0])
        work = copy.deepcopy(self.core)
        work.update(reference_kind='government_mining_clearing_weak_context', context_pixels=420,
                    verified_accident_negative=True)
        self.assertFalse(requested_use(work, 'accident_negative')[0])

    def test_unaccepted_draft_or_changed_asset_is_rejected(self):
        self.core['existing_scope_admitted'] = False
        self.assertFalse(requested_use(self.core, 'accident_positive_core')[0])
        self.core['existing_scope_admitted'] = True
        self.core['integrity_passed'] = False
        self.assertFalse(requested_use(self.core, 'accident_positive_core')[0])

    def test_renamed_spatial_crop_still_overlaps_protected_extent(self):
        self.assertTrue(bounds_intersect([1, 1, 2, 2], [0, 0, 3, 3]))
        self.assertFalse(bounds_intersect([4, 4, 5, 5], [0, 0, 3, 3]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
