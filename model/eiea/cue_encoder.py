"""
EIEA cue encoder: Mamba-2 interpreter + structured input encoders + projection.

Architecture (paper §3, "Efficient Image-Embodied Adapter"):
  image (CLIP/SigLIP) ──┐
  depth (SigLIP)      ──┤→ [proj → Mamba-2 → φ_R] → 10 × d_VLM cue tokens
  3D boxes            ──┤
  BEV grid            ──┘
  question text       ──┘

The 10 cue tokens are injected at designated <TOR> positions in the frozen
VLM's input embeddings via masked_scatter.

Sub-modules:
  MIDIBackbone       Mamba-2 sequential encoder (pretrained SSR checkpoint)
  BoxEncoder         class embedding + MLP → box tokens
  BEVEncoder         CNN → spatial tokens
  CueProjector       d_mamba → d_VLM linear projection (3584 → 2048)
"""

import torch
import torch.nn as nn


class MIDIBackbone(nn.Module):
    """Mamba-2 sequential encoder.

    Takes projected image/depth embeddings, optional structured-cue tokens,
    and question text tokens; outputs hidden states at TOR positions.

    This is a thin wrapper around a pretrained `MambaForCausalLM` checkpoint
    (state-spaces/mamba-130m-hf). The image/depth/text projections are
    fine-tuned during EIEA alignment training.
    """

    def __init__(self, mamba_model, hidden_size: int = 768):
        super().__init__()
        self.mamba = mamba_model
        self.hidden_size = hidden_size
        self.image_proj = nn.Sequential(
            nn.Linear(1024, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size))
        self.depth_proj = nn.Sequential(
            nn.Linear(1152, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size))

    def forward(self, image_embeds, depth_embeds, cue_tokens, text_ids,
                tor_token_id: int):
        """Args:
            image_embeds: [1, N_img, d_clip]  CLIP last hidden states
            depth_embeds: [1, N_dep, d_sig]   SigLIP last hidden states
            cue_tokens:   [N_cue, hidden]     optional structured-cue tokens
                                          (boxes / BEV / etc.; may be empty)
            text_ids:     [1, L_text]         tokenised question (+ TOR markers)
            tor_token_id: int                token id of <TOR> in text_ids

        Returns:
            hidden: [1, N_img + N_dep + N_cue + L_text, hidden]  full Mamba output
            tor_hidden: [N_tor, hidden]  hidden states at <TOR> positions
        """
        ie = self.image_proj(image_embeds)
        de = self.depth_proj(depth_embeds)
        te = self.mamba.get_input_embeddings()(text_ids)
        parts = [ie, de]
        if cue_tokens is not None and cue_tokens.shape[0] > 0:
            parts.append(cue_tokens.unsqueeze(0))
        parts.append(te)
        seq = torch.cat(parts, dim=1)
        mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=seq.device)
        out = self.mamba(inputs_embeds=seq, attention_mask=mask,
                         output_hidden_states=True)
        hidden = out.hidden_states[-1]
        tor_hidden = hidden[:, ie.size(1) + de.size(1):, :][:, text_ids.eq(0)[:, 0].cumsum(0) < 0]  # placeholder
        # Extract TOR-position hidden states.
        tor_start = ie.size(1) + de.size(1) + (cue_tokens.shape[0] if cue_tokens is not None and cue_tokens.shape[0] > 0 else 0)
        tor_end = tor_start + (text_ids == tor_token_id).sum().item()
        tor_hidden = hidden[:, tor_start:tor_end, :]
        return hidden, tor_hidden


class BoxEncoder(nn.Module):
    """Encode 3D boxes into tokens for the Mamba sequence.

    Each box is [class_id, cx, cy, cz, length, width, height, yaw, vx, vy, dist].
    """

    def __init__(self, num_classes: int = 11, hidden_size: int = 768):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, 16)
        self.mlp = nn.Sequential(
            nn.Linear(16 + 10, 512), nn.GELU(),
            nn.Linear(512, hidden_size))

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """Args:
            boxes: [N, 11] float tensor (class_id + 10 numeric features)
        Returns:
            tokens: [N, hidden_size]
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, self.mlp[-1].out_features,
                               device=boxes.device, dtype=boxes.dtype)
        cls = self.class_emb(boxes[:, 0].long())
        feats = boxes[:, 1:11]
        return self.mlp(torch.cat([cls, feats], -1))


class BEVEncoder(nn.Module):
    """Encode a BEV grid into spatial tokens for the Mamba sequence.

    Input is a multi-channel BEV raster (e.g., drivable area, lane, road
    divider, ped crossing, walkway; or RGB point-cloud BEV).
    """

    def __init__(self, in_channels: int = 5, hidden_size: int = 768,
                 grid_size: int = 200):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)))
        self.proj = nn.Linear(128, hidden_size)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """Args:
            bev: [C, H, W] BEV raster
        Returns:
            tokens: [64, hidden_size]
        """
        f = self.conv(bev.unsqueeze(0))          # [1, 128, 8, 8]
        f = f.flatten(2).transpose(1, 2)          # [1, 64, 128]
        return self.proj(f).squeeze(0)            # [64, hidden_size]


class CueProjector(nn.Module):
    """Project Mamba hidden states (d_mamba) to VLM embedding space (d_VLM).

    Follows SSR's rationale-to-semantic design: RMSNorm → Linear → GELU → Linear.
    """

    def __init__(self, d_mamba: int = 768, d_vlm: int = 2048):
        super().__init__()
        self.rms = nn.RMSNorm(d_mamba, eps=1e-6)
        self.fc1 = nn.Linear(d_mamba, d_vlm)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_vlm, d_vlm)

    def forward(self, tor_hidden: torch.Tensor) -> torch.Tensor:
        """Args:
            tor_hidden: [N_tor, d_mamba]
        Returns:
            cue_tokens: [N_tor, d_VLM]
        """
        return self.fc2(self.act(self.fc1(self.rms(tor_hidden))))


class EIEACueEncoder(nn.Module):
    """Full EIEA cue encoder: combines all sub-encoders into 10 cue tokens.

    Training: end-to-end with answer cross-entropy through a frozen VLM.
    Inference: cue tokens are injected at <TOR> positions via masked_scatter.
    """

    def __init__(self, mamba_model, num_classes: int = 11,
                 bev_channels: int = 5, hidden_size: int = 768,
                 d_vlm: int = 2048, num_tor: int = 10):
        super().__init__()
        self.midi = MIDIBackbone(mamba_model, hidden_size)
        self.box_enc = BoxEncoder(num_classes, hidden_size)
        self.bev_enc = BEVEncoder(bev_channels, hidden_size)
        self.proj = CueProjector(hidden_size, d_vlm)
        self.num_tor = num_tor
        self.d_vlm = d_vlm
        self.hidden_size = hidden_size
        self.queries = nn.Parameter(torch.randn(num_tor, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)

    def forward(self, image_embeds, depth_embeds, boxes, bev, text_ids,
                tor_token_id: int):
        """Args:
            image_embeds: [1, N_img, d_clip]
            depth_embeds: [1, N_dep, d_sig]
            boxes: [N, 11] or empty
            bev: [C, H, W] or None
            text_ids: [1, L_text] question token ids (without TOR markers)
            tor_token_id: int

        Returns:
            cue_tokens: [1, num_tor, d_VLM]
        """
        ie = self.midi.image_proj(image_embeds)
        de = self.midi.depth_proj(depth_embeds)
        parts = [ie, de]
        if boxes is not None and boxes.shape[0] > 0:
            parts.append(self.box_enc(boxes).unsqueeze(0))
        if bev is not None and bev.abs().sum() > 0:
            parts.append(self.bev_enc(bev).unsqueeze(0))
        te = self.midi.mamba.get_input_embeddings()(text_ids)
        # Insert 10 learnable TOR queries at the end (before text tokens).
        tor_q = self.queries.unsqueeze(0).to(te.dtype)
        parts.append(tor_q)
        parts.append(te)
        seq = torch.cat(parts, dim=1)
        mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=seq.device)
        out = self.midi.mamba(inputs_embeds=seq, attention_mask=mask,
                              output_hidden_states=True)
        hidden = out.hidden_states[-1]
        # Extract the 10 TOR query positions.
        tor_start = ie.size(1) + de.size(1) + (boxes.shape[0] if boxes is not None and boxes.shape[0] > 0 else 0) + self.num_tor
        tor_start = sum(p.shape[1] for p in parts[:-1])  # all except text
        tor_hidden = hidden[:, tor_start:tor_start + self.num_tor, :]
        cue = self.midi.tor_proj(tor_hidden.squeeze(0) if tor_hidden.dim() == 3 else tor_hidden)
        return cue.unsqueeze(0)  # [1, num_tor, d_VLM]
