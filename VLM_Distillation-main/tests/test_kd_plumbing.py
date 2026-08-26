import unittest
from types import SimpleNamespace


class KDPlumbingTests(unittest.TestCase):
    def test_default_empty_kd_loss_type_uses_ce_only_criterion(self):
        from src.criterions import CEOnlyCriterion, build_criterion

        criterion = build_criterion(SimpleNamespace(kd_loss_type=""))

        self.assertIsInstance(criterion, CEOnlyCriterion)

    def test_emkd_reads_weight_arguments(self):
        from src.criterions.em_kd import EMKDCriterion

        args = SimpleNamespace(
            em_kd_alpha=0.7,
            em_kd_beta=0.11,
            em_kd_gamma=3.5,
            em_kd_temperature=2.25,
        )

        criterion = EMKDCriterion(args)

        self.assertEqual(criterion.alpha, 0.7)
        self.assertEqual(criterion.beta, 0.11)
        self.assertEqual(criterion.gamma, 3.5)
        self.assertEqual(criterion.temperature, 2.25)

    def test_joint_alias_builds_unit_aligned_criterion(self):
        from src.criterions import UnitAlignedDistillationCriterion, build_criterion

        for alias in ("joint", "unit_aligned", "unit_aligned_distillation"):
            with self.subTest(alias=alias):
                criterion = build_criterion(SimpleNamespace(kd_loss_type=alias))
                self.assertIsInstance(criterion, UnitAlignedDistillationCriterion)

    def test_joint_mode_enables_sre_pooler_in_collator(self):
        from src.data.dataset import VlmDistillDataCollator

        collator = VlmDistillDataCollator(
            student_processor=object(),
            teacher_processor=object(),
            data_args=SimpleNamespace(kd_loss_type="joint"),
            model_args=SimpleNamespace(teacher_model_name="teacher"),
        )

        self.assertTrue(collator.use_sre_pooler)

    def test_scva_reads_cluster_arguments(self):
        from src.criterions.scva import SCVACriterion

        args = SimpleNamespace(
            scva_alpha=0.6,
            scva_weight=2.0,
            scva_n_clusters=24,
            scva_kmeans_iters=15,
            scva_attention_layer=-2,
            scva_min_vision_tokens=8,
        )
        criterion = SCVACriterion(args)
        self.assertEqual(criterion.alpha, 0.6)
        self.assertEqual(criterion.weight, 2.0)
        self.assertEqual(criterion.n_clusters, 24)
        self.assertEqual(criterion.kmeans_iters, 15)
        self.assertEqual(criterion.attention_layer, -2)
        self.assertEqual(criterion.min_vision_tokens, 8)

    def test_cgkd_reads_arguments(self):
        from src.criterions.cgkd import CGKDCriterion

        args = SimpleNamespace(cgkd_alpha=0.3, cgkd_weight=4.0, cgkd_temperature=1.5)
        criterion = CGKDCriterion(args)
        self.assertEqual(criterion.alpha, 0.3)
        self.assertEqual(criterion.weight, 4.0)
        self.assertEqual(criterion.temperature, 1.5)

    def test_scva_cgkd_joint_builds(self):
        from src.criterions import SCVACGKDCriterion, build_criterion

        for alias in ("scva_cgkd", "draft"):
            with self.subTest(alias=alias):
                args = SimpleNamespace(
                    kd_loss_type=alias,
                    # SCVA + CGKD sub-criterion defaults
                    scva_alpha=0.5, scva_weight=1.0, scva_n_clusters=16,
                    scva_kmeans_iters=10, scva_attention_layer=-1, scva_min_vision_tokens=4,
                    cgkd_alpha=0.5, cgkd_weight=1.0, cgkd_temperature=1.0,
                    # joint formula coefficients (draft notation)
                    scva_cgkd_ce_weight=1.0, scva_cgkd_lambda_v=0.7, scva_cgkd_lambda_g=0.4,
                )
                criterion = build_criterion(args)
                self.assertIsInstance(criterion, SCVACGKDCriterion)
                self.assertEqual(criterion.lambda_v, 0.7)
                self.assertEqual(criterion.lambda_g, 0.4)
                self.assertEqual(criterion.ce_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
