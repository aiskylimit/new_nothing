import unittest
from types import SimpleNamespace

import torch


class KDPlumbingTests(unittest.TestCase):
    def test_mcw_pairwise_costs_match_broadcast_implementations_and_gradients(self):
        from src.criterions.mcw_kd import pairwise_cosine_cost, pairwise_kl_cost

        torch.manual_seed(11)
        teacher_logits = torch.randn(4, 7)
        student_logits = torch.randn(5, 7)
        teacher_probs = torch.softmax(teacher_logits, dim=-1).requires_grad_()
        student_probs = torch.softmax(student_logits, dim=-1).requires_grad_()

        actual_kl = pairwise_kl_cost(teacher_probs, student_probs)
        expected_kl = (
            teacher_probs.unsqueeze(1)
            * torch.log(
                (
                    teacher_probs.unsqueeze(1)
                    / student_probs.unsqueeze(0).clamp_min(1e-9)
                ).clamp_min(1e-9)
            )
        ).sum(dim=-1)
        torch.testing.assert_close(actual_kl, expected_kl, atol=1e-6, rtol=1e-5)

        actual_kl.sum().backward(retain_graph=True)
        actual_teacher_grad = teacher_probs.grad.detach().clone()
        actual_student_grad = student_probs.grad.detach().clone()
        teacher_probs.grad = None
        student_probs.grad = None
        expected_kl.sum().backward()
        torch.testing.assert_close(teacher_probs.grad, actual_teacher_grad, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(student_probs.grad, actual_student_grad, atol=2e-6, rtol=1e-5)

        # Preserve the original inner ratio clamp and its gradient even in the
        # extreme low-probability branch.
        extreme_teacher = torch.tensor(
            [[1e-12, 1.0 - 1e-12], [0.5, 0.5]], requires_grad=True
        )
        extreme_student = torch.tensor(
            [[0.9, 0.1], [1e-12, 1.0 - 1e-12]], requires_grad=True
        )
        actual_extreme = pairwise_kl_cost(extreme_teacher, extreme_student, teacher_chunk=1)
        expected_extreme = (
            extreme_teacher.unsqueeze(1)
            * torch.log(
                (
                    extreme_teacher.unsqueeze(1)
                    / extreme_student.unsqueeze(0).clamp_min(1e-9)
                ).clamp_min(1e-9)
            )
        ).sum(dim=-1)
        torch.testing.assert_close(actual_extreme, expected_extreme, atol=0, rtol=0)

        actual_extreme.sum().backward(retain_graph=True)
        actual_teacher_grad = extreme_teacher.grad.detach().clone()
        actual_student_grad = extreme_student.grad.detach().clone()
        extreme_teacher.grad = None
        extreme_student.grad = None
        expected_extreme.sum().backward()
        torch.testing.assert_close(extreme_teacher.grad, actual_teacher_grad, atol=0, rtol=0)
        torch.testing.assert_close(extreme_student.grad, actual_student_grad, atol=0, rtol=0)

        teacher = torch.randn(4, 9, requires_grad=True)
        student = torch.randn(5, 9, requires_grad=True)
        actual_cosine = pairwise_cosine_cost(teacher, student)
        expected_cosine = 1.0 - torch.nn.functional.cosine_similarity(
            teacher.unsqueeze(1), student.unsqueeze(0), dim=-1
        )
        torch.testing.assert_close(actual_cosine, expected_cosine, atol=1e-6, rtol=1e-5)

        actual_cosine.sum().backward(retain_graph=True)
        actual_teacher_grad = teacher.grad.detach().clone()
        actual_student_grad = student.grad.detach().clone()
        teacher.grad = None
        student.grad = None
        expected_cosine.sum().backward()
        torch.testing.assert_close(teacher.grad, actual_teacher_grad, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(student.grad, actual_student_grad, atol=2e-6, rtol=1e-5)

    def test_mcw_context_window_matches_loop_and_gradient(self):
        from src.criterions.mcw_kd import context_window_mean

        torch.manual_seed(13)
        seq = torch.randn(9, 6, requires_grad=True)
        actual = context_window_mean(seq, window=2)
        expected = torch.stack(
            [seq[max(index - 2, 0) : min(index + 3, seq.shape[0])].mean(dim=0) for index in range(seq.shape[0])]
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

        actual.square().sum().backward(retain_graph=True)
        actual_grad = seq.grad.detach().clone()
        seq.grad = None
        expected.square().sum().backward()
        torch.testing.assert_close(seq.grad, actual_grad, atol=2e-6, rtol=1e-5)

    def test_mcw_chunked_vocab_operations_match_full_operations(self):
        from src.criterions.mcw_kd import max_softmax_probability_sum, topk_values_chunked

        torch.manual_seed(19)
        logits = torch.randn(2, 11, 23, requires_grad=True)
        mask = torch.rand(2, 11) > 0.3

        actual_topk = topk_values_chunked(logits, k=6, sequence_chunk=4)
        expected_topk = torch.topk(logits.float(), 6, dim=-1).values
        torch.testing.assert_close(actual_topk, expected_topk)

        actual_topk.sum().backward(retain_graph=True)
        actual_grad = logits.grad.detach().clone()
        logits.grad = None
        expected_topk.sum().backward()
        torch.testing.assert_close(logits.grad, actual_grad)

        actual_probability = max_softmax_probability_sum(logits, mask, sequence_chunk=4)
        expected_probability = (
            torch.softmax(logits.detach().float(), dim=-1).amax(dim=-1) * mask.float()
        ).sum()
        torch.testing.assert_close(actual_probability, expected_probability, atol=1e-6, rtol=1e-5)

    def test_mcw_optimized_dtw_keeps_original_alignment(self):
        from src.criterions.mcw_kd import dist_fn_edit, dtw_alignment

        def reference_alignment(series_1, series_2):
            matrix = [[float("inf")] * (len(series_2) + 1) for _ in range(len(series_1) + 1)]
            matrix[0][0] = 0.0
            for i, value_1 in enumerate(series_1):
                for j, value_2 in enumerate(series_2):
                    matrix[i + 1][j + 1] = float(dist_fn_edit(value_1, value_2)) + min(
                        matrix[i][j + 1], matrix[i + 1][j], matrix[i][j]
                    )
            matrix = [row[1:] for row in matrix[1:]]
            i, j = len(series_1) - 1, len(series_2) - 1
            aligned = []
            while i > 0 or j > 0:
                aligned.append((i, j))
                options = [
                    matrix[i - 1][j - 1] if i > 0 and j > 0 else float("inf"),
                    matrix[i - 1][j] if i > 0 else float("inf"),
                    matrix[i][j - 1] if j > 0 else float("inf"),
                ]
                move = min(range(3), key=lambda idx: options[idx])
                if move == 0:
                    i -= 1
                    j -= 1
                elif move == 1:
                    i -= 1
                else:
                    j -= 1
            aligned.append((0, 0))
            return aligned

        cases = [
            (["a", "b", "a", "ccc"], ["a", "a", "c"]),
            (["same", "same", "x"], ["same", "y", "x", "x"]),
            (["aa", "bb"], ["cc", "dd"]),
        ]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                self.assertEqual(dtw_alignment(left, right), reference_alignment(left, right))

    def test_etp_optimized_loop_matches_original_output_and_gradient(self):
        from src.criterions.etp import ETP

        def reference_etp(cost, alpha=0.1, threshold=1e-9, max_iter=100, epsilon=1e-9):
            rows, cols = cost.shape
            a = cost.new_full((rows, 1), 1.0 / rows)
            b = cost.new_full((cols, 1), 1.0 / cols)
            u = cost.new_full((rows, 1), 1.0 / rows)
            kernel = torch.exp(-cost.float() * alpha).clamp_min(epsilon)
            err = cost.new_tensor(float("inf"))
            step = 0
            while err > threshold and step < max_iter:
                v = b / (kernel.t().matmul(u) + epsilon)
                u = a / (kernel.matmul(v) + epsilon)
                step += 1
                if step % 50 == 1:
                    marginal = v * kernel.t().matmul(u)
                    err = torch.norm(torch.sum(torch.abs(marginal - b), dim=0), p=float("inf"))
            transport = u * (kernel * v.t())
            return torch.sum(transport * cost), transport

        torch.manual_seed(17)
        cost = torch.rand(7, 5, requires_grad=True)
        actual_loss, actual_transport = ETP(max_iter=100)(cost)
        expected_loss, expected_transport = reference_etp(cost)
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(actual_transport, expected_transport)

        actual_loss.backward(retain_graph=True)
        actual_grad = cost.grad.detach().clone()
        cost.grad = None
        expected_loss.backward()
        torch.testing.assert_close(cost.grad, actual_grad, atol=1e-6, rtol=1e-5)

    def test_mcw_forward_still_backpropagates_all_trainable_branches(self):
        from src.criterions.mcw_kd import MCWKDCriterion

        class TinyModel(torch.nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

            def get_input_embeddings(self):
                return self.embed

            def get_output_embeddings(self):
                return self.lm_head

            def forward(self, input_ids, labels=None, **_kwargs):
                hidden = self.embed(input_ids)
                return SimpleNamespace(
                    logits=self.lm_head(hidden),
                    hidden_states=(hidden,),
                    text_feature_mask=torch.ones_like(input_ids, dtype=torch.bool),
                )

        class TinyTokenizer:
            @staticmethod
            def convert_ids_to_tokens(ids, **_kwargs):
                return [f"token-{token_id}" for token_id in ids]

        student = TinyModel(vocab_size=13, hidden_size=4)
        teacher = TinyModel(vocab_size=13, hidden_size=6)
        teacher.requires_grad_(False)
        projectors = torch.nn.ModuleDict(
            {
                "query": torch.nn.Linear(8, 12),
                "s2t": torch.nn.Linear(4, 6),
                "t2s": torch.nn.Linear(6, 4),
            }
        )
        processor = SimpleNamespace(tokenizer=TinyTokenizer())
        distiller = SimpleNamespace(
            student=student,
            teacher=teacher,
            projectors=projectors,
            get_student_processor=lambda: processor,
            get_teacher_processor=lambda: processor,
        )
        criterion = MCWKDCriterion(
            SimpleNamespace(
                kd_rate=0.5,
                kd_objective="forward_kl",
                kd_temperature=1.0,
                teacher_temperature=1.0,
                top_k_vocab=3,
                mcw_tau_seq=2.0,
                mcw_window_size=2,
                mcw_ot_logits_rate=1.0,
                mcw_ot_hidden_rate=1.0,
                mcw_sinkhorn_alpha=0.1,
                mcw_sinkhorn_iter=2,
                max_steps=1,
            )
        )
        student_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        teacher_ids = torch.tensor([[1, 2, 7, 3, 4, 5, 6]])
        batch = {
            "student_inputs": {
                "input_ids": student_ids,
                "labels": torch.tensor([[-100, -100, 3, 4, 5, 6]]),
            },
            "teacher_inputs": {
                "input_ids": teacher_ids,
                "labels": torch.tensor([[-100, -100, -100, 3, 4, 5, 6]]),
            },
        }

        output = criterion(distiller, batch)
        output["loss"].backward()

        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertIsNotNone(student.embed.weight.grad)
        for projector in projectors.values():
            self.assertIsNotNone(projector.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_dwa_cosine_cost_matmul_matches_pairwise_cosine(self):
        from src.criterions.dwa_kd import cosine_cost_matrices

        torch.manual_seed(7)
        student = torch.randn(5, 8)
        teacher = torch.randn(3, 8)

        cross, student_cost, teacher_cost = cosine_cost_matrices(student, teacher)

        expected_cross = 1.0 - torch.nn.functional.cosine_similarity(
            student.unsqueeze(1), teacher.unsqueeze(0), dim=-1
        )
        expected_student = 1.0 - torch.nn.functional.cosine_similarity(
            student.unsqueeze(1), student.unsqueeze(0), dim=-1
        )
        expected_teacher = 1.0 - torch.nn.functional.cosine_similarity(
            teacher.unsqueeze(1), teacher.unsqueeze(0), dim=-1
        )
        torch.testing.assert_close(cross, expected_cross)
        torch.testing.assert_close(student_cost, expected_student)
        torch.testing.assert_close(teacher_cost, expected_teacher)

    def test_attention_outputs_follow_criterion_requirements(self):
        from src.distiller import Distiller

        distiller = object.__new__(Distiller)
        attention_free = (
            "ce_only",
            "default",
            "emkd",
            "cgkd",
            "dwa_kd",
            "dskd_v2",
            "mcw_kd",
        )
        attention_required = (
            "sre",
            "scva",
            "scva_cgkd",
            "joint",
            "unit_aligned",
        )

        for kd_loss_type in attention_free:
            with self.subTest(kd_loss_type=kd_loss_type):
                distiller.training_args = SimpleNamespace(kd_loss_type=kd_loss_type)
                self.assertFalse(distiller._needs_attention_outputs())

        for kd_loss_type in attention_required:
            with self.subTest(kd_loss_type=kd_loss_type):
                distiller.training_args = SimpleNamespace(kd_loss_type=kd_loss_type)
                self.assertTrue(distiller._needs_attention_outputs())

    def test_vlm_forward_respects_disabled_attention_outputs(self):
        from src.model.model import VLMModel

        class RecordingEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.config = SimpleNamespace()
                self.forward_kwargs = None

            def forward(self, **kwargs):
                self.forward_kwargs = kwargs
                return kwargs

        encoder = RecordingEncoder()
        model = VLMModel(encoder, output_attentions=False)
        model(input_ids=torch.ones(1, 2, dtype=torch.long))

        self.assertFalse(encoder.forward_kwargs["output_attentions"])
        self.assertTrue(encoder.forward_kwargs["output_hidden_states"])

    def test_dskd_forward_backpropagates_through_student_and_both_projectors(self):
        from src.criterions.dskd_v2 import DSKDv2Criterion

        class TinyModel(torch.nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

            def get_output_embeddings(self):
                return self.lm_head

            def forward(self, input_ids, labels=None, **_kwargs):
                hidden = self.embed(input_ids)
                logits = self.lm_head(hidden)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                return SimpleNamespace(
                    loss=loss,
                    logits=logits,
                    hidden_states=(hidden,),
                    text_feature_mask=torch.ones_like(input_ids, dtype=torch.bool),
                )

        student = TinyModel(vocab_size=11, hidden_size=4)
        teacher = TinyModel(vocab_size=11, hidden_size=6)
        teacher.requires_grad_(False)
        projectors = torch.nn.ModuleDict(
            {
                "t2s": torch.nn.Sequential(torch.nn.Linear(6, 4)),
                "s2t": torch.nn.Sequential(torch.nn.Linear(4, 6)),
            }
        )
        distiller = SimpleNamespace(student=student, teacher=teacher, projectors=projectors)
        criterion = DSKDv2Criterion(
            SimpleNamespace(
                kd_rate=0.5,
                kd_objective="forward_kl",
                kd_temperature=1.0,
                teacher_temperature=1.0,
                only_stu_kd=False,
                only_tea_kd=False,
                init_s2t_projector=False,
                t2s_agreement=1.0,
                label_smoothing=0.0,
            )
        )
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[-100, -100, 3, 4, 5]])
        batch = {
            "student_inputs": {"input_ids": input_ids, "labels": labels},
            "teacher_inputs": {"input_ids": input_ids, "labels": labels},
        }

        output = criterion(distiller, batch)
        output["loss"].backward()

        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertIsNotNone(student.embed.weight.grad)
        self.assertIsNotNone(projectors["t2s"][0].weight.grad)
        self.assertIsNotNone(projectors["s2t"][0].weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_dskd_topk_alias_overrides_default_minus_one(self):
        from src.criterions.dskd_v2 import resolve_topk_vocab
        from src.distiller import _resolve_dskd_topk_vocab

        args = SimpleNamespace(dskd_topk_vocab=-1, topk_vocab=2048)
        self.assertEqual(resolve_topk_vocab(args), 2048)
        self.assertEqual(_resolve_dskd_topk_vocab(args), 2048)

        args = SimpleNamespace(dskd_topk_vocab=512, topk_vocab=2048)
        self.assertEqual(resolve_topk_vocab(args), 512)
        self.assertEqual(_resolve_dskd_topk_vocab(args), 512)

    def test_dskd_dynamic_s2t_initialization_freezes_unused_module(self):
        from src.distiller import Distiller

        class TinyHeadModel(torch.nn.Module):
            def __init__(self, hidden_size, vocab_size):
                super().__init__()
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

            def get_output_embeddings(self):
                return self.lm_head

        class TinyTokenizer:
            def get_vocab(self):
                return {f"token-{index}": index for index in range(7)}

        distiller = Distiller.__new__(Distiller)
        torch.nn.Module.__init__(distiller)
        distiller.training_args = SimpleNamespace(
            kd_loss_type="dskd_v2",
            init_t2s_projector=True,
            init_s2t_projector=True,
            dskd_topk_vocab=-1,
            topk_vocab=3,
        )
        distiller.student = TinyHeadModel(hidden_size=4, vocab_size=7)
        distiller.teacher = TinyHeadModel(hidden_size=6, vocab_size=7)
        distiller.projectors = torch.nn.ModuleDict(
            {
                "t2s": torch.nn.Sequential(torch.nn.Linear(6, 4)),
                "s2t": torch.nn.Sequential(torch.nn.Linear(4, 6)),
            }
        )
        processor = SimpleNamespace(tokenizer=TinyTokenizer())
        distiller._student_processor = processor
        distiller._teacher_processor = processor

        distiller.init_dskd_projectors_if_needed()

        self.assertEqual(tuple(distiller.part_teacher_head_pinv.shape), (3, 6))
        self.assertFalse(any(parameter.requires_grad for parameter in distiller.projectors["s2t"].parameters()))

    def test_dskd_auto_creates_projectors_without_json(self):
        from src.distiller import Distiller

        distiller = Distiller.__new__(Distiller)
        torch.nn.Module.__init__(distiller)
        distiller.student_hidden_dim = 4
        distiller.teacher_hidden_dim = 6
        distiller.training_args = SimpleNamespace(kd_loss_type="dskd_v2", teacher_layer_mapping=[])
        distiller.model_args = SimpleNamespace(
            projector_config_path="/path/that/must/not/be/opened.json",
            proj_dim=3,
        )

        distiller.set_projector()

        self.assertEqual(set(distiller.projectors), {"t2s", "s2t"})
        self.assertEqual(tuple(distiller.projectors["t2s"][0].weight.shape), (4, 6))
        self.assertEqual(tuple(distiller.projectors["s2t"][0].weight.shape), (6, 4))

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
