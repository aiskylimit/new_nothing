"""Project-owned VLMEvalKit dataset adapters."""

from collections import defaultdict
from numbers import Number

import numpy as np

from vlmeval.dataset.image_yorn import ImageYORNDataset
from vlmeval.dataset.image_mcq import ImageMCQDataset
from vlmeval.dataset.image_vqa import ImageVQADataset
from vlmeval.dataset.utils.yorn import YOrN_Extraction
from vlmeval.smp import d2df, dump, get_intermediate_file_path, load


class _LimitedRowsMixin:
    """Slice after upstream has expanded shared base64-image references."""

    def __init__(self, dataset, max_samples=100, **kwargs):
        self._max_samples = int(max_samples)
        if self._max_samples <= 0:
            raise ValueError("max_samples must be positive")
        super().__init__(dataset=dataset, **kwargs)

    def post_build(self, dataset):
        super().post_build(dataset)
        self.data = self.data.iloc[:self._max_samples].reset_index(drop=True).copy()


class LimitedImageMCQDataset(_LimitedRowsMixin, ImageMCQDataset):
    """Small deterministic prefix of an upstream MCQ dataset."""


class LimitedImageVQADataset(_LimitedRowsMixin, ImageVQADataset):
    """Small deterministic prefix of an upstream VQA dataset."""


class MMEPerceptionDataset(ImageYORNDataset):
    """Evaluate and report only the official MME perception categories."""

    PERCEPTION_CATEGORIES = {
        "OCR", "artwork", "celebrity", "color", "count", "existence",
        "landmark", "position", "posters", "scene",
    }

    def post_build(self, dataset):
        if dataset != "MME":
            raise ValueError(f"MMEPerceptionDataset cannot load {dataset}")
        self.data = self.data[self.data["category"].isin(self.PERCEPTION_CATEGORIES)].copy()

    def evaluate(self, eval_file, **judge_kwargs):
        """Use MME's official ACC + ACC+ formula on perception only."""
        if judge_kwargs.get("model", "exact_matching") != "exact_matching":
            raise ValueError("MME-Perception in this offline suite supports exact_matching only")
        data = load(eval_file)
        data["prediction"] = [str(value) for value in data["prediction"]]
        data["extracted"] = [YOrN_Extraction(value) for value in data["prediction"]]
        data["score"] = data["answer"] == data["extracted"]
        storage = get_intermediate_file_path(eval_file, "_auxmatch")
        dump(data, storage)

        grouped = defaultdict(lambda: defaultdict(list))
        for _, row in data.iterrows():
            grouped[row["category"]][row["image_path"]].append(bool(row["score"]))

        category_scores = {}
        for category, image_answers in grouped.items():
            answers = [answer for values in image_answers.values() for answer in values]
            paired = [values[0] * values[1] for values in image_answers.values()]
            category_scores[category] = np.mean(answers) * 100 + np.mean(paired) * 100
        result = {"perception": sum(category_scores.values()), **category_scores}
        result = d2df(result)
        dump(result, get_intermediate_file_path(eval_file, "_score", "csv"))
        return result

    @classmethod
    def report_primary_metric(cls, metrics):
        value = metrics.get("perception") if isinstance(metrics, dict) else None
        return {"perception": float(value)} if isinstance(value, Number) else {}
