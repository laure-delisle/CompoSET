"""MMVP-VLM — batched runner (drop-in for compo_eval/mmvp.py).

135 image pairs from MMVP/MMVP_VLM. Sequential pairing of Questions.csv.
Pulls files via hf_hub_download (HF auto-derived dataset discards questions).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from ..models import VLMScorer


MMVP_HF = "MMVP/MMVP_VLM"


def run_mmvp(
    scorer: VLMScorer,
    img_bs: int = 32,
    txt_bs: int = 64,
) -> pd.DataFrame:
    try:
        qcsv = hf_hub_download(repo_id=MMVP_HF, filename="Questions.csv",
                               repo_type="dataset")
    except Exception as e:
        print(f"[mmvp] WARN: Questions.csv download failed: {repr(e)[:120]}")
        return pd.DataFrame()

    with open(qcsv) as f:
        statements = {int(r["Question ID"]): (r.get("Type", ""), r["Statement"])
                      for r in csv.DictReader(f)}

    try:
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=MMVP_HF, repo_type="dataset")
    except Exception as e:
        print(f"[mmvp] WARN: repo file list failed: {repr(e)[:120]}")
        return pd.DataFrame()

    path_by_id: dict[int, str] = {}
    for fpath in repo_files:
        if fpath.endswith(".jpg"):
            try:
                path_by_id[int(Path(fpath).stem)] = fpath
            except ValueError:
                continue

    ids = sorted(statements.keys())
    pairs_meta = []
    images = []
    texts = []
    for i in range(0, len(ids) - 1, 2):
        id_a, id_b = ids[i], ids[i + 1]
        if id_a not in path_by_id or id_b not in path_by_id:
            continue
        type_a, stmt_a = statements[id_a]
        _, stmt_b = statements[id_b]
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
        pairs_meta.append({
            "id": f"{id_a}_{id_b}",
            "type": type_a,
            "stmt_a": stmt_a,
            "stmt_b": stmt_b,
        })
        images.append(img_a)
        images.append(img_b)
        texts.append(stmt_a)
        texts.append(stmt_b)

    if not pairs_meta:
        return pd.DataFrame()

    print(f"MMVP [{scorer.model_key}]: encoding {len(images)} images "
          f"and {len(texts)} texts (batched)")
    img_emb = scorer.encode_images(images, batch_size=img_bs)
    txt_emb = scorer.encode_texts(texts, batch_size=txt_bs)

    rows = []
    for k, m in enumerate(pairs_meta):
        sim_aa = float((img_emb[2 * k]     @ txt_emb[2 * k]).item())
        sim_ab = float((img_emb[2 * k]     @ txt_emb[2 * k + 1]).item())
        sim_ba = float((img_emb[2 * k + 1] @ txt_emb[2 * k]).item())
        sim_bb = float((img_emb[2 * k + 1] @ txt_emb[2 * k + 1]).item())
        i2t = int(sim_aa > sim_ab and sim_bb > sim_ba)
        t2i = int(sim_aa > sim_ba and sim_bb > sim_ab)
        grp = int(i2t and t2i)
        rows.append({**m, "sim_aa": sim_aa, "sim_ab": sim_ab,
                     "sim_ba": sim_ba, "sim_bb": sim_bb,
                     "i2t_score": i2t, "t2i_score": t2i, "group_score": grp})
    return pd.DataFrame(rows)
