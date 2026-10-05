# Differentiable Augmentation
# Modified from
#   https://github.com/mit-han-lab/data-efficient-gans

import torch

# ----------------------------------------------------------------------------
# stateless ops. each returns the FULLY augmented batch (per-sample random
# parameters); the per-sample probability gating is handled by DiffAug below.
# only translation and horizontal flip are kept. images are assumed
# channels-first [B, C, H, W] and roughly unit-scaled (e.g. [-1, 1]).
# ----------------------------------------------------------------------------


def _wrap_indices(idx, size, mode):
    # map (possibly out-of-range) integer indices back into [0, size-1].
    if mode == "circular":
        return torch.remainder(idx, size)
    if mode == "reflect":  # reflect_101 (no edge repeat), like F.pad(mode="reflect")
        if size == 1:
            return torch.zeros_like(idx)
        period = 2 * (size - 1)
        idx = torch.remainder(idx, period)
        return torch.where(idx >= size, period - idx, idx)
    raise ValueError(f"unknown translation padding_mode: {mode}")


def rand_translation(x, ratio=0.125, padding_mode="zeros"):
    # per-sample integer shift up to +-ratio of each spatial dim. padding_mode
    # fills the vacated strip:
    #   "zeros"    - black fill (classic DiffAugment; leaks a border tell into a
    #                fake-only distribution critic).
    #   "reflect"  - mirror the edge (no border artifact; leak-safe).
    #   "circular" - wrap around (no border artifact; leak-safe, wrap seam).
    B, _, H, W = x.shape
    shift_h, shift_w = int(H * ratio + 0.5), int(W * ratio + 0.5)
    tx = torch.randint(-shift_h, shift_h + 1, size=[B, 1, 1], device=x.device)
    ty = torch.randint(-shift_w, shift_w + 1, size=[B, 1, 1], device=x.device)
    grid_b, grid_h, grid_w = torch.meshgrid(
        torch.arange(B, dtype=torch.long, device=x.device),
        torch.arange(H, dtype=torch.long, device=x.device),
        torch.arange(W, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    src_h = grid_h + tx  # [B, H, W] source coords each output pixel reads from
    src_w = grid_w + ty
    if padding_mode == "zeros":
        valid = (src_h >= 0) & (src_h < H) & (src_w >= 0) & (src_w < W)
        src_h, src_w = src_h.clamp(0, H - 1), src_w.clamp(0, W - 1)
    else:
        src_h = _wrap_indices(src_h, H, padding_mode)
        src_w = _wrap_indices(src_w, W, padding_mode)
        valid = None
    x = (
        x.permute(0, 2, 3, 1)
        .contiguous()[grid_b, src_h, src_w]  # [B, H, W, C]
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    if valid is not None:
        x = x * valid.unsqueeze(1).to(x.dtype)
    return x


def rand_hflip(x):
    # horizontal flip of every sample; per-sample gating handled by DiffAug
    return torch.flip(x, dims=[-1])


class DiffAug:
    """
    per-module differentiable augmentation
    """

    def __init__(
        self,
        translation_prob=0.0,
        flip_prob=0.0,
        translation_ratio=0.125,
        translation_padding="zeros",
    ):
        # (name, callable(x) -> aug'd x, per-sample prob); ordered pipeline.
        self.pipeline = [
            (
                "translation",
                lambda x: rand_translation(x, translation_ratio, translation_padding),
                translation_prob,
            ),
            ("flip", rand_hflip, flip_prob),
        ]

    def __str__(self):
        active = ", ".join(
            f"{name}(p={prob:g})" for name, _, prob in self.pipeline if prob > 0
        )
        return f"DiffAug({active or 'identity'})"

    __repr__ = __str__

    @staticmethod
    def _blend(x, x_aug, prob):
        # per-sample Bernoulli(prob) selection between augmented and original,
        # differentiable w.r.t. both branches.
        if prob >= 1.0:
            return x_aug
        mask = (torch.rand(x.size(0), device=x.device) < prob).view(-1, 1, 1, 1)
        return torch.where(mask, x_aug, x)

    def aug(self, x):
        _dtype = x.dtype
        x = x.float()
        for _, fn, prob in self.pipeline:
            if prob <= 0.0:
                continue
            x = self._blend(x, fn(x), prob)
        return x.to(_dtype)
