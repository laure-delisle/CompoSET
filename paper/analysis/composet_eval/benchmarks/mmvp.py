"""MMVP loader (Tong et al., CVPR 2024) — MMVP-VLM variant.

135 image pairs from CLIP-blind contrasts. The HF release at
``MMVP/MMVP_VLM`` ships images organized by category folders plus a
``Questions.csv`` with the per-image statement. The HF auto-derived dataset
loses the question text (it builds a folder→label classification dataset),
so we ignore ``load_dataset`` and pull files directly via ``hf_hub_download``.

Pair scheme: 270 question rows in CSV order; pair k = (img_2k-1, img_2k) ↔
(statement_2k-1, statement_2k). Each pair is one CLIP-blind contrast — two
visually-distinct images that CLIP confuses, with two statements that
differ on the contrasted attribute.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


MMVP_HF = "MMVP/MMVP_VLM"


class MMVPLoader(BenchmarkLoader):
    name = "mmvp"
    same_scene = "no"
    has_image_pair = True
    has_text_pair = True

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        try:
            qcsv = hf_hub_download(repo_id=MMVP_HF, filename="Questions.csv",
                                   repo_type="dataset")
        except Exception as e:
            print(f"[mmvp] WARN: Questions.csv download failed: {repr(e)[:120]}")
            return

        with open(qcsv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Build {question_id (int) -> (type, statement)}.
        statements: dict[int, tuple[str, str]] = {}
        for r in rows:
            qid = int(r["Question ID"])
            statements[qid] = (r.get("Type", ""), r["Statement"])

        # Map question_id -> repo path. Categories aren't sequential by ID,
        # so list the repo files once and index by filename stem.
        try:
            api = HfApi()
            repo_files = api.list_repo_files(repo_id=MMVP_HF, repo_type="dataset")
        except Exception as e:
            print(f"[mmvp] WARN: repo file list failed: {repr(e)[:120]}")
            return

        path_by_id: dict[int, str] = {}
        for fpath in repo_files:
            if not fpath.endswith(".jpg"):
                continue
            try:
                stem = Path(fpath).stem  # "241"
                path_by_id[int(stem)] = fpath
            except ValueError:
                continue

        n = 0
        # Pair sequentially: (1,2), (3,4), ..., (269, 270).
        ids = sorted(statements.keys())
        for i in range(0, len(ids) - 1, 2):
            id_a, id_b = ids[i], ids[i + 1]
            if id_a not in path_by_id or id_b not in path_by_id:
                continue
            type_a, stmt_a = statements[id_a]
            type_b, stmt_b = statements[id_b]
            try:
                p_a = hf_hub_download(repo_id=MMVP_HF, filename=path_by_id[id_a],
                                      repo_type="dataset")
                p_b = hf_hub_download(repo_id=MMVP_HF, filename=path_by_id[id_b],
                                      repo_type="dataset")
                img_a = Image.open(p_a).convert("RGB")
                img_b = Image.open(p_b).convert("RGB")
            except Exception as e:
                print(f"[mmvp] WARN: image fetch failed for ids {id_a},{id_b}: "
                      f"{repr(e)[:100]}")
                continue
            yield BenchmarkPair(
                benchmark=self.name,
                pair_id=f"{id_a}_{id_b}",
                image_a=img_a,
                image_b=img_b,
                caption_positive=stmt_a,
                caption_negative=stmt_b,
                same_scene=self.same_scene,
                has_image_pair=True,
                has_text_pair=True,
                extras={"id_a": id_a, "id_b": id_b, "type": type_a},
            )
            n += 1
            if limit is not None and n >= limit:
                return
