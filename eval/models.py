"""Model registry and scoring wrapper for vision-language models.

Add new models by extending the MODELS dict.

Custom-weight models (negclip, dreamlip, clove) need their weights downloaded
manually and placed under ``$COMPOSET_WEIGHTS_DIR`` (default
``~/.cache/composet/weights``). See the per-entry comments in MODELS for
sources.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import torch
import open_clip
from PIL import Image

# Patch: transformers 5.0 removed batch_encode_plus from slow tokenizers.
# open_clip's HFTokenizer (used by SigLIP) still calls it.
import transformers
for _tok_name in ("T5Tokenizer", "GemmaTokenizer",
                  "XLMRobertaTokenizer", "XLMRobertaTokenizerFast"):
    _tok = getattr(transformers, _tok_name, None)
    if _tok is None:
        continue
    try:
        if not hasattr(_tok, "batch_encode_plus"):
            _tok.batch_encode_plus = _tok.__call__
    except ImportError:
        # Tokenizer class is a stub (missing backend like sentencepiece); skip.
        pass

# Patch: transformers 5.0 renamed AutoModelForVision2Seq -> AutoModelForImageTextToText.
# GME-Qwen2-VL's `custom_st.py` still imports the old name.
if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(transformers, "AutoModelForImageTextToText"):
    transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText


WEIGHTS_DIR = Path(
    os.environ.get("COMPOSET_WEIGHTS_DIR", str(Path.home() / ".cache/composet/weights"))
)


# ── Registry ────────────────────────────────────────────────────────────────
# Each entry maps a short alias to backend-specific constructor args.
MODELS: dict[str, dict] = {
    # --- Phase 0: contrastive baselines ---
    "clip-b32": {"model_name": "ViT-B-32", "pretrained": "openai"},
    "clip-l14": {"model_name": "ViT-L-14", "pretrained": "openai"},
    "clip-b16-openai": {"model_name": "ViT-B-16", "pretrained": "openai"},
    "clip-b16-laion400m": {"model_name": "ViT-B-16", "pretrained": "laion400m_e32"},
    "clip-b16-laion2b": {"model_name": "ViT-B-16", "pretrained": "laion2b_s34b_b88k"},
    "siglip-b16": {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"},
    "siglip2-b16": {"model_name": "ViT-B-16-SigLIP2", "pretrained": "webli"},

    # --- Phase 1a: scale-matched CLIP / SigLIP / SigLIP2 ---
    "clip-l14-laion2b": {"model_name": "ViT-L-14", "pretrained": "laion2b_s32b_b82k"},
    "clip-h14-laion2b": {"model_name": "ViT-H-14", "pretrained": "laion2b_s32b_b79k"},
    "siglip-l16-384": {"model_name": "ViT-L-16-SigLIP-384", "pretrained": "webli"},
    "siglip2-l16-384": {"model_name": "ViT-L-16-SigLIP2-384", "pretrained": "webli"},
    "siglip-so400m-384": {"model_name": "ViT-SO400M-14-SigLIP-384", "pretrained": "webli"},
    "siglip2-so400m-378": {"model_name": "ViT-SO400M-14-SigLIP2-378", "pretrained": "webli"},

    # --- Phase 1b: data-curation / new-family baselines (open_clip native) ---
    "dfn-b16": {"model_name": "ViT-B-16-quickgelu", "pretrained": "dfn2b"},
    "dfn-h14": {"model_name": "ViT-H-14-quickgelu", "pretrained": "dfn5b"},
    "eva02-b16": {"model_name": "EVA02-B-16", "pretrained": "merged2b_s8b_b131k"},
    "eva02-l14": {"model_name": "EVA02-L-14", "pretrained": "merged2b_s4b_b131k"},
    "metaclip2-h14": {"model_name": "ViT-H-14-worldwide-quickgelu", "pretrained": "metaclip2_worldwide"},

    # --- Phase 2: CLIP-arch with custom weights (local .pt / .bin) ---
    # NegCLIP: open_clip-formatted .bin. Place at $COMPOSET_WEIGHTS_DIR/negclip-b32/open_clip_pytorch_model.bin
    # Source: https://huggingface.co/HuggingFaceM4/negclip
    "negclip-b32": {
        "model_name": "ViT-B-32",
        "pretrained": str(WEIGHTS_DIR / "negclip-b32" / "open_clip_pytorch_model.bin"),
    },
    # DreamLIP: place merged30M ckpt at $COMPOSET_WEIGHTS_DIR/dreamlip-b16-merged30m.pt
    # Source: https://github.com/zyf0619sjtu/DreamLIP
    "dreamlip-b16-merged30m": {
        "model_name": "ViT-B-16",
        "pretrained": None,
        "weights_path": str(WEIGHTS_DIR / "dreamlip-b16-merged30m.pt"),
        "state_dict_renamer": "strip_prefix",
    },
    # CLoVe: place ckpt at $COMPOSET_WEIGHTS_DIR/clove/clove_without_patching.pt
    # Source: https://github.com/sallymo3rgan/clove
    "clove-b32": {
        "model_name": "ViT-B-32",
        "pretrained": None,
        "weights_path": str(WEIGHTS_DIR / "clove" / "clove_without_patching.pt"),
        "state_dict_renamer": "strip_prefix",
        "create_kwargs": {"force_context_length": 64},
    },

    # --- Phase 3: non-open_clip backends (single-vector cosine dual-encoders) ---
    # Jina v4: Qwen2.5-VL-3B-based multimodal dual encoder (2048-dim).
    "jina-embeddings-v4": {
        "backend": "jina",
        "hf_repo": "jinaai/jina-embeddings-v4",
    },
}


def _strip_state_dict_prefix(sd: dict) -> dict:
    """Strip DDP / Lightning wrapper prefixes from a state_dict."""
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


_RENAMERS = {
    "strip_prefix": _strip_state_dict_prefix,
}


class VLMScorer:
    """Image-text similarity scorer.

    Default backend is open_clip; alternate backends can be selected via
    ``cfg["backend"]`` in the MODELS registry (e.g. ``"jina"``).
    """

    def __init__(self, model_key: str, device: str = "cuda"):
        if model_key not in MODELS:
            raise ValueError(
                f"Unknown model key {model_key!r}. "
                f"Available: {list(MODELS.keys())}"
            )
        cfg = MODELS[model_key]
        self.model_key = model_key
        self.device = device
        self.backend = cfg.get("backend", "open_clip")

        if self.backend == "jina":
            from transformers import AutoModel
            self.jina_model = AutoModel.from_pretrained(
                cfg["hf_repo"], trust_remote_code=True, torch_dtype=torch.float16,
            ).to(device).eval()
            return

        # open_clip backend (default)
        create_kwargs = cfg.get("create_kwargs", {})
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg["model_name"], pretrained=cfg.get("pretrained"), device=device,
            **create_kwargs,
        )
        tokenizer_kwargs = {}
        if "force_context_length" in create_kwargs:
            tokenizer_kwargs["context_length"] = create_kwargs["force_context_length"]
        self.tokenizer = open_clip.get_tokenizer(cfg["model_name"], **tokenizer_kwargs)

        # Optional: load local weights with a renamer (e.g. DreamLIP, CLoVe).
        weights_path = cfg.get("weights_path")
        if weights_path:
            wp = Path(weights_path)
            if not wp.exists():
                raise FileNotFoundError(
                    f"Weights file not found: {wp}\n"
                    f"For {model_key!r}, see comment in models.py for source URL "
                    f"and place the file at {wp} (or set COMPOSET_WEIGHTS_DIR)."
                )
            sd = torch.load(wp, map_location=device, weights_only=False)
            for inner in ("state_dict", "model", "module"):
                if isinstance(sd, dict) and inner in sd and isinstance(sd[inner], dict):
                    sd = sd[inner]
                    break
            renamer = _RENAMERS.get(cfg.get("state_dict_renamer"))
            if renamer is not None:
                sd = renamer(sd)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            print(f"[{model_key}] load_state_dict: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"  missing[:5]: {missing[:5]}")
            if unexpected:
                print(f"  unexpected[:5]: {unexpected[:5]}")

        self.model.eval()

    # ── public API ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def sim(
        self,
        image: Union[str, Path, Image.Image],
        texts: list[str],
    ) -> list[float]:
        """Cosine similarity between one image and each text."""
        if self.backend == "jina":
            import torch.nn.functional as F
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            ie = self.jina_model.encode_image(images=[image], task="retrieval")
            te = self.jina_model.encode_text(
                texts=list(texts), task="retrieval", prompt_name="passage",
            )
            ie_t = torch.stack([torch.as_tensor(x) for x in ie]).float()
            te_t = torch.stack([torch.as_tensor(x) for x in te]).float()
            sims = (F.normalize(ie_t, dim=-1) @ F.normalize(te_t, dim=-1).T).squeeze(0)
            return sims.tolist()

        img_feat = self._encode_image(image)
        txt_feat = self._encode_text(texts)
        sims = (img_feat @ txt_feat.T).squeeze(0)
        return sims.tolist()

    # ── batched API (used by batched runners) ───────────────────────────────
    # Returns L2-normalized embeddings as CPU tensors so downstream matmul
    # stays off the GPU and memory stays bounded.

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 32) -> torch.Tensor:
        """Batched image encoding.

        Accepts any iterable yielding PIL images / str paths / Path objects.
        Uses lazy iteration so generators are safe (bounds memory at
        batch_size x image_size).
        """
        import gc
        from itertools import islice
        import torch.nn.functional as F

        it = iter(images)
        out = []
        while True:
            batch = list(islice(it, batch_size))
            if not batch:
                break
            pil = []
            for im in batch:
                if isinstance(im, (str, Path)):
                    im = Image.open(im)
                pil.append(im.convert("RGB"))

            if self.backend == "jina":
                ie = self.jina_model.encode_image(images=pil, task="retrieval")
                emb = torch.stack([torch.as_tensor(x) for x in ie]).float()
                emb = F.normalize(emb, dim=-1)
            else:
                tensor = torch.stack(
                    [self.preprocess(im) for im in pil]
                ).to(self.device)
                emb = self.model.encode_image(tensor, normalize=True)

            out.append(emb.detach().cpu())
            del pil, batch, emb
            gc.collect()
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: list, batch_size: int = 64) -> torch.Tensor:
        import torch.nn.functional as F
        out = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])

            if self.backend == "jina":
                te = self.jina_model.encode_text(
                    texts=batch, task="retrieval", prompt_name="passage",
                )
                emb = torch.stack([torch.as_tensor(x) for x in te]).float()
                emb = F.normalize(emb, dim=-1)
            else:
                tokens = self.tokenizer(batch).to(self.device)
                emb = self.model.encode_text(tokens, normalize=True)

            out.append(emb.detach().cpu())
        return torch.cat(out, dim=0)

    # ── internals ───────────────────────────────────────────────────────────

    def _encode_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model.encode_image(tensor, normalize=True)

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        return self.model.encode_text(tokens, normalize=True)
