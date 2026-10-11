"""The idea encoder alone, on CPU, for analysis on the laptop: ``Encoder(adapter).encode(texts)``.

Same tokenizer, LoRA config and mean pooling as ``remote-encoder.py``'s ``SonarAE.encode``; only the encoder half of
the adapter is loaded, in float32. Vectors come back L2-normalized.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

ENC = ("cointegrated/SONAR_200_text_encoder", "29f2eac73636d1548c2cb227dbfb2772b31ebc4f")
ATT = ["q_proj", "k_proj", "v_proj", "out_proj"]


class Encoder:
    def __init__(self, adapter: str | None = None, r: int = 16):
        self.tok = AutoTokenizer.from_pretrained(ENC[0], revision=ENC[1])
        self.tok.src_lang = "eng_Latn"
        enc = M2M100Encoder.from_pretrained(ENC[0], revision=ENC[1], torch_dtype=torch.float32)
        if adapter:
            enc = get_peft_model(enc, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=ATT + ["fc1", "fc2"]))
            sd = {k[4:]: v.float() for k, v in torch.load(adapter, map_location="cpu").items() if k.startswith("enc.")}
            res = enc.load_state_dict(sd, strict=False)
            assert not res.unexpected_keys, res.unexpected_keys
        self.enc = enc.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch: int = 64, max_len: int = 96) -> torch.Tensor:
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), batch):
            b = self.tok([texts[i] for i in order[s : s + batch]], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            h = self.enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1))
        v = torch.cat(out)
        res = torch.empty_like(v)
        res[order] = v
        return res
