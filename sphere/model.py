from functools import partial
from typing import Optional

import logging
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from sphere.layers import vector_rms_norm, grid_rms_norm, stratified_unit_radii_ddp
from sphere.transformer import Transformer
from sphere.utils import vector_compute_magnitude
from sphere.encoder import DinoV, SigLIP2
from cli_utils import get_device_type

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("G1_LOG_LEVEL", "INFO").upper())

"""
variables
"""


PRETRAINED_ENCODERS = [
    # -----
    "dinov2-small",
    "dinov2-with-registers-small",
    "dinov2-base",
    "dinov2-with-registers-base",
    "dinov2-large",
    "dinov2-with-registers-large",
    # -----
    "dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3-vits16-pretrain-lvd1689m",
    "dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-vitl16-pretrain-lvd1689m",
    # -----
    "siglip2-base-patch16-256",
    "siglip2-base-patch16-384",
    "siglip2-base-patch16-512",
    "siglip2-large-patch16-256",
    "siglip2-large-patch16-384",
    "siglip2-large-patch16-512",
    "siglip2-so400m-patch16-256",
    "siglip2-so400m-patch16-384",
    "siglip2-so400m-patch16-512",
]

"""
functions
"""


def rotate_away_from(z, z_ref, deg, normalize):
    zc, zu = z.float(), z_ref.float()
    cos = F.cosine_similarity(zc.flatten(1), zu.flatten(1), dim=1)
    cos = cos.view(-1, *([1] * (z.ndim - 1)))
    # component of -z_ref orthogonal to z: the tangent direction pointing away
    t_raw = cos * zc - zu
    t_norm = t_raw.flatten(1).norm(dim=1).view_as(cos)
    t = normalize(t_raw).float()
    beta = torch.tensor(math.radians(deg), device=z.device, dtype=torch.float32)
    out = torch.cos(beta) * zc + torch.sin(beta) * t
    out = torch.where(t_norm > 1e-6, out, zc)
    return out.to(z.dtype)


def slerp(z_a, z_b, t, dim):
    """
    spherical interpolation between z_a and z_b

    z_a, z_b : same shape, same norm over dim
    t        : fraction in [0, 1]
    dim      : dims the angle is measured over (all non-batch dims for global
               spherify, the channel dim for local)
    out      : same shape as z_a
    """
    a, b = z_a.float(), z_b.float()
    cos = (a * b).sum(dim=dim, keepdim=True) / (
        a.norm(dim=dim, keepdim=True) * b.norm(dim=dim, keepdim=True) + 1e-8
    )
    omega = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
    so = torch.sin(omega)
    out = (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b
    return out.to(z_a.dtype)


"""
classes
"""


class G1(nn.Module):  # first gear

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        vit_enc_model_size: str = "base",
        vit_dec_model_size: str = "base",
        enc_train_last_n_blocks: int = 0,
        enc_train_norm_ls_layers: bool = False,
        enc_train_final_norm: bool = False,
        drop_residual_path_prob: float = 0.0,
        token_channels: int = 128,
        num_classes: int = 0,
        in_context_size: int = 0,
        pixel_head_type: str = "linear",
        pixel_head_use_tanh: bool = False,
        halve_model_size: bool = False,
        freeze_encoder: bool = False,
        spherify_model: bool = False,
        spherify_mode: str = "global",
        load_pretrained_enc: str = None,
        pretrained_encoder_interpolate_tokens: bool = False,
        pretrained_encoder_interpolate_pos_embed: bool = False,
        pretrained_encoder_fix_compression_ratio: bool = False,
        use_latent_consistency: bool = False,
        noise_sigma_max_angle: float = 90.0,
        latent_consistency_use_ema: bool = False,
        latent_consistency_freeze_encoder: bool = False,
        use_angle_condition: bool = False,
        angle_condition_mode: str = "adaln",
        sdpa_mode: Optional[str] = None,
        use_qk_norm: bool = False,
    ):
        super().__init__()

        self.encoder = Transformer(
            input_size=input_size,
            patch_size=patch_size,
            model_size=vit_enc_model_size,
            model_type="encoder",
            token_chns=token_channels,
            num_classes=0,  # G1 uses unconditional encoder
            in_context_size=in_context_size,
            halve_model_size=halve_model_size,
            drop_residual_path_prob=drop_residual_path_prob,
            sdpa_mode=sdpa_mode,
            use_qk_norm=use_qk_norm,
        )
        if freeze_encoder:
            self.encoder.eval().requires_grad_(False)

        if load_pretrained_enc is not None:
            assert load_pretrained_enc in PRETRAINED_ENCODERS
            num_tokens = self.encoder.num_tokens

            if "siglip2" in load_pretrained_enc:
                encoder_cls = SigLIP2
            elif "dinov" in load_pretrained_enc:
                encoder_cls = DinoV
            else:
                raise ValueError(f"Unknown pretrained encoder: {load_pretrained_enc}")

            self.encoder = encoder_cls(
                model_name=load_pretrained_enc,
                normalize=not enc_train_final_norm,
                freeze=freeze_encoder,
                train_last_n_blocks=enc_train_last_n_blocks,
                train_final_norm=enc_train_final_norm,
                train_norm_ls=enc_train_norm_ls_layers,
                interpolate_num_tokens_to=(
                    num_tokens if pretrained_encoder_interpolate_tokens else None
                ),
                interpolate_pos_embed=pretrained_encoder_interpolate_pos_embed,
            )

            # adjust modules for sphere encoder
            self.encoder.use_modulation = False
            token_chns = self.encoder.token_chns

            if pretrained_encoder_fix_compression_ratio:
                if token_channels > token_chns:
                    logger.warning(
                        f"forced latent width {token_channels} > encoder width "
                        f"{token_chns}: this is an UP-projection out of a "
                        f"{token_chns}-wide bottleneck, so the requested "
                        f"compression ratio is nominal -- the real bottleneck "
                        f"stays at "
                        f"{3 * input_size**2 / (num_tokens * token_chns):.2f}x"
                    )
                self.encoder.out = nn.Linear(token_chns, token_channels)
            else:
                token_channels = token_chns

            logger.info(
                f"pretrained encoder latent: {num_tokens} x {token_channels} "
                f"(backbone width {token_chns}), compression "
                f"{3 * input_size**2 / (num_tokens * token_channels):.2f}x"
            )

        if freeze_encoder:
            assert hasattr(self.encoder, "out")
            if hasattr(self.encoder, "out"):  # but keep the adapter trainable
                self.encoder.out.train().requires_grad_(True)

        self.decoder = Transformer(
            input_size=input_size,
            patch_size=patch_size,
            model_size=vit_dec_model_size,
            model_type="decoder",
            token_chns=token_channels,
            num_classes=num_classes,
            in_context_size=in_context_size,
            halve_model_size=halve_model_size,
            spherify_model=spherify_model,
            pixel_head_type=pixel_head_type,
            pixel_head_use_tanh=pixel_head_use_tanh,
            drop_residual_path_prob=drop_residual_path_prob,
            use_angle_cond=use_angle_condition,
            angle_cond_max_deg=noise_sigma_max_angle,
            angle_cond_mode=angle_condition_mode,
            sdpa_mode=sdpa_mode,
            use_qk_norm=use_qk_norm,
        )

        self.num_classes = num_classes
        self.latent_shape = self.decoder.latent_shape  # [1, N, D]
        self.num_tokens = self.latent_shape[1]
        self.use_modulation = self.encoder.use_modulation or self.decoder.use_modulation

        self.spherify_mode = spherify_mode
        spherify_fn = {"global": vector_rms_norm, "local": grid_rms_norm}[
            self.spherify_mode
        ]
        self.f = partial(spherify_fn, zero_mean=False)

        self.lat_con_use_ema = latent_consistency_use_ema
        # freeze the 2nd (re-encode) pass encoder with detached live weights
        self.lat_con_freeze_enc = latent_consistency_freeze_encoder
        self.use_lat_con = use_latent_consistency

        # differentiable pixel augmentations for latent consistency loss
        self.score_augs = None

        # training noise angles are uniform in [0, max_angle_deg]
        self.max_angle_deg = noise_sigma_max_angle

        # decoder noise-angle conditioning (see sphere.layers.AngleEmbedder)
        self.use_angle_cond = use_angle_condition
        if self.use_angle_cond:
            logger.info(
                f"angle conditioning: decoder, "
                f"max_angle={noise_sigma_max_angle} deg"
            )

        self.log_dict = {}
        self.ddp_rank = None
        self.ddp_world_size = None
        self.step_seed = None
        self.cached_noise = None
        self._last_sched_key = None

    """
    functions for training
    """

    def _dec_angle(self, n, deg, device):
        if not self.use_angle_cond:
            return None
        return torch.full(
            (n, 1, 1), np.deg2rad(float(deg)), device=device, dtype=torch.float32
        )

    def frozen_encoder_func(self, x, y=None):
        params = {
            name: param.detach() for name, param in self.encoder.named_parameters()
        }
        buffers = dict(self.encoder.named_buffers())
        return functional_call(self.encoder, (params, buffers), (x, y))

    @torch.no_grad()
    def _log_angle_embed(self, alpha_cond):
        a_embed = self.decoder.alpha_embedder(alpha_cond).squeeze(1).float()
        a_norm = a_embed.norm(dim=-1).mean()
        a_std = a_embed.std(dim=0, correction=0).mean()
        self.log_dict["avg_angle_embed_norm"] = a_norm.detach()
        self.log_dict["avg_angle_embed_alpha_std"] = a_std.detach()
        self.log_dict["angle_embed_alpha_frac"] = (a_std / (a_norm + 1e-8)).detach()

    def forward(self, x, y=None, ema_model=None):
        # always empty: G1 extracts no intermediate hidden states any more
        hidden_states = {}
        latents = {}

        # encode
        z = self.encoder(x, y)  # [B, N, D]
        # backbone CLS of the clean encode (pretrained encoders only)
        cls_clean = getattr(self.encoder, "last_cls", None)

        # log stats
        mag_z = vector_compute_magnitude(z)
        self.log_dict["avg_mag_z"] = mag_z.mean().detach()
        self.log_dict["std_mag_z"] = mag_z.std(correction=0).detach()

        # cast to fp32 for geometric computations
        _dtype = z.dtype
        z = z.float()

        # spherify latents
        z = self.f(z)
        z_clean = z.clone()  # [B, N, D]
        latents["z_clean"] = z_clean
        if cls_clean is not None:
            latents["cls_clean"] = cls_clean.float()

        # get noise
        e = torch.randn_like(z)
        e = self.f(e)

        # get angles: stratified U[0, 1] across all ranks, scaled to max angle
        r = stratified_unit_radii_ddp(
            size=z.shape,
            rank=self.ddp_rank,
            world_size=self.ddp_world_size,
            step_seed=self.step_seed,
            shuffle=True,  # shuffle to avoid rank bias
            including_zero=True,  # include the identity mapping (angle 0)
            device=z.device,
        )
        alpha_NOISY = math.radians(self.max_angle_deg) * r

        # get noisy latents
        z_NOISY = torch.cos(alpha_NOISY) * z + torch.sin(alpha_NOISY) * e
        z_NOISY = z_NOISY.to(_dtype)  # cast back (alpha_* are in fp32)

        norms_NOISY = z_NOISY.flatten(1).norm(dim=1)
        self.log_dict["avg_norm_z_NOISY"] = norms_NOISY.mean().detach()

        # decode latents
        alpha_cond = alpha_NOISY.detach() if self.use_angle_cond else None
        x = self.decoder(z_NOISY, y, alpha=alpha_cond)

        if alpha_cond is not None:
            self._log_angle_embed(alpha_cond)

        pixels = {
            "x_NOISY": x,
            "alpha_NOISY": alpha_NOISY.detach(),
        }

        # latent consistency loss part
        if self.use_lat_con:
            if self.score_augs is not None:
                x = self.score_augs.aug(x)

            if ema_model is not None and self.lat_con_use_ema:
                enc_mod = ema_model.module.encoder
                v = enc_mod(x, y)
            else:
                enc_mod = self.encoder  # frozen_encoder_func runs this module
                enc_fn = (
                    self.frozen_encoder_func
                    if self.lat_con_freeze_enc
                    else self.encoder
                )
                v = enc_fn(x, y)

            v = self.f(v)

            # backbone CLS of the re-encode
            cls_v = getattr(enc_mod, "last_cls", None)

            latents["v_NOISY"] = v
            if cls_v is not None:
                latents["cls_NOISY"] = cls_v.float()

        return pixels, hidden_states, latents

    """
    functions for generation
    """

    def spherify(
        self,
        z,
        angle_deg,
        cache_noise=False,
    ):
        z = self.f(z)
        dtype, B = z.dtype, z.shape[0]

        if angle_deg <= 0.0:
            return z, self._dec_angle(B, 0.0, z.device)

        e = self.cached_noise if cache_noise else None
        if e is not None and e.shape != z.shape:
            logger.warning(
                f"[spherify] cached noise {tuple(e.shape)} does not match latent "
                f"{tuple(z.shape)}: redrawing"
            )
            e = None
        if e is None:
            e = self.f(torch.randn_like(z))
            if cache_noise:
                self.cached_noise = e

        alpha = torch.full(
            (B, 1, 1),
            np.deg2rad(float(angle_deg)),
            device=z.device,
            dtype=torch.float32,
        )

        if logger.isEnabledFor(logging.DEBUG):
            cos_ez = F.cosine_similarity(
                e.float().flatten(1), z.float().flatten(1), dim=1
            )
            ang_ez = torch.rad2deg(torch.acos(cos_ez)).mean().item()
            logger.debug(
                f"[spherify] target={angle_deg:.2f}deg"
                f" applied={torch.rad2deg(alpha).mean().item():.2f}deg"
                f" cos(e,z)={cos_ez.mean().item():+.4f}"
                f" ang(e,z)={ang_ez:.2f}deg"
            )

        z = torch.cos(alpha) * z + torch.sin(alpha) * e
        return z.to(dtype), (alpha if self.use_angle_cond else None)

    @torch.no_grad()
    def generate(
        self,
        batch_size,
        y=None,
        cfg=0.0,
        cfg_position="angle",
        forward_steps=1,
        sampling_init_angle=85.0,
        sampling_loop_angle=None,
        cache_sampling_noise=False,
        return_step_images=False,
        device=get_device_type(),
    ):
        assert (
            cfg_position == "angle"
        ), f"only angle cfg is supported, got {cfg_position}"
        self.cached_noise = None  # clean the cache

        do_ang_cfg = cfg > 0.0
        ang_cfg_deg = cfg if do_ang_cfg else 0.0

        e = torch.randn(batch_size, *self.latent_shape[1:]).to(device)
        if cache_sampling_noise:
            self.cached_noise = self.f(e)

        if self.use_modulation:
            if y is None:
                y = torch.randint(
                    low=0,
                    high=self.decoder.num_classes,
                    size=(batch_size,),
                    device=device,
                )
            y_uncond = torch.full_like(y, self.num_classes) if do_ang_cfg else None
        else:
            y, y_uncond = None, None

        alpha = self._dec_angle(
            batch_size, min(self.max_angle_deg, sampling_init_angle), device
        )
        z = self.f(e)
        x = self.decoder(z, y, alpha=alpha)

        if do_ang_cfg:
            x_uncond = self.decoder(z, y_uncond, alpha=alpha)

        h = x.clone()
        h = torch.clamp(h * 0.5 + 0.5, 0, 1)

        if forward_steps == 1:
            if return_step_images:
                return h, h, [h]
            return h, h

        if return_step_images:
            step_images = [h]

        T = forward_steps
        if sampling_loop_angle is None:
            sampling_loop_angle = sampling_init_angle
        angles = [min(self.max_angle_deg, sampling_loop_angle)] * (T - 1)

        # log once per distinct setting, not once per batch
        sched_key = tuple(angles)
        if getattr(self, "_last_sched_key", None) != sched_key:
            self._last_sched_key = sched_key
            logger.info(
                f"[generate] sampling angles: {[round(a, 4) for a in angles]} deg"
            )

        for step, angle_deg in enumerate(angles, start=1):
            z = self.encoder(x, y)

            if do_ang_cfg:
                z_uncond = self.f(self.encoder(x_uncond, y_uncond))
                z = self.f(z)
                z = rotate_away_from(z, z_uncond, ang_cfg_deg, self.f)

            z, alpha = self.spherify(
                z,
                angle_deg,
                cache_noise=cache_sampling_noise,
            )
            x = self.decoder(z, y, alpha=alpha)

            if do_ang_cfg and step < len(angles):
                z_uncond, _ = self.spherify(
                    z_uncond,
                    angle_deg,
                    cache_noise=cache_sampling_noise,
                )
                x_uncond = self.decoder(z_uncond, y_uncond, alpha=alpha)

            if return_step_images:
                step_images.append(torch.clamp(x * 0.5 + 0.5, 0, 1))

        x = torch.clamp(x * 0.5 + 0.5, 0, 1)
        if return_step_images:
            return h, x, step_images
        return h, x

    """
    functions for reconstuction
    """

    @torch.no_grad()
    def encode(self, x, y=None):
        if self.num_classes > 0 and y is None:
            y = torch.full((x.shape[0],), self.num_classes)
            y = y.to(device=x.device, dtype=torch.long)  # null class embedding
        z = self.encoder(x, y)
        return z

    @torch.no_grad()
    def decode(self, z, y=None, alpha_deg=0.0):
        if self.num_classes > 0 and y is None:
            y = torch.full((z.shape[0],), self.num_classes)
            y = y.to(device=z.device, dtype=torch.long)  # null class embedding
        alpha = self._dec_angle(z.shape[0], alpha_deg, z.device)
        x = self.decoder(z, y, alpha=alpha)
        return x * 0.5 + 0.5

    @torch.no_grad()
    def reconstruct(
        self,
        x,
        y=None,
        noise_scaler=1.0,
        sampling=False,
        continue_sampling=False,
        forward_steps=2,
        cache_sampling_noise=False,
        sampling_max_angle=0.0,
    ):
        if self.num_classes > 0 and y is None:
            y = torch.full((x.shape[0],), self.num_classes)
            y = y.to(device=x.device, dtype=torch.long)  # null class embedding
        z = self.encoder(x, y)
        max_deg = min(self.max_angle_deg, sampling_max_angle)
        z, alpha = self.spherify(
            z,
            noise_scaler * max_deg if sampling else 0.0,
            cache_noise=cache_sampling_noise,
        )
        x = self.decoder(z, y, alpha=alpha)
        x = torch.clamp(x * 0.5 + 0.5, 0, 1)

        if continue_sampling:
            x = x * 2.0 - 1.0

            for _ in range(forward_steps - 1):
                z = self.encoder(x, y)
                z, alpha = self.spherify(
                    z,
                    min(self.max_angle_deg, 89.0),
                    cache_noise=cache_sampling_noise,
                )
                x = self.decoder(z, y, alpha=alpha)

            x = torch.clamp(x * 0.5 + 0.5, 0, 1)

        return x

    """
    functions for editing
    """

    @torch.no_grad()
    def interpolate(
        self,
        y_a,
        y_b,
        num_interp=8,
        forward_steps=1,
        init_angle=85.0,
        sampling_loop_angle=None,
        cache_sampling_noise=True,
        class_interp="lerp",
        fix_latent=False,
        device=get_device_type(),
    ):
        """
        decode num_interp points on the slerp path between two random points e_a
        and e_b. every point e_t is decoded exactly like the first step of
        generate (alpha = init_angle); with forward_steps > 1 the usual refinement
        loop follows for each point, rotating toward e_t itself when the noise
        cache is on, with the same interpolated conditioning

        y_a, y_b     : [B] class ids
        class_interp : how the class conditioning is interpolated,
                       lerp (linear mix of the two class embeddings) |
                       slerp (spherical mix of the two class embeddings) |
                       hard (y_a for t < 0.5, y_b otherwise, no mixing)
        fix_latent   : draw a single point and keep it for every column, so only
                       the class conditioning moves along the path
        out          : list of num_interp tensors [B, 3, H, W] in [0, 1], from the
                       y_a end to the y_b end
        """
        assert class_interp in ["lerp", "slerp", "hard"]
        B = y_a.shape[0]
        y_a = y_a.to(device=device, dtype=torch.long)
        y_b = y_b.to(device=device, dtype=torch.long)

        # class embeddings of the endpoints, [B, 1, D]; None for an
        # unconditional decoder (then cond_embed stays None and y is unused)
        emb = getattr(self.decoder, "y_embedder", None)
        c_a = emb(y_a, False) if emb is not None else None
        c_b = emb(y_b, False) if emb is not None else None

        def cond_at(t):
            """
            (y_t, cond_embed_t) for the decoder at fraction t
            """
            y_t = y_a if t < 0.5 else y_b
            if c_a is None or class_interp == "hard":
                return y_t, None
            if class_interp == "lerp":
                c_t = (1.0 - t) * c_a + t * c_b
            else:
                c_t = slerp(c_a, c_b, t, (2,))
            return y_t, c_t.to(c_a.dtype)

        slerp_dim = (
            tuple(range(1, len(self.latent_shape)))
            if self.spherify_mode == "global"
            else (2,)
        )

        e_a = self.f(torch.randn(B, *self.latent_shape[1:], device=device))
        if fix_latent:
            assert (
                class_interp != "hard"
            ), "fix_latent with hard labels gives only two distinct images"
            e_b = e_a
        else:
            e_b = self.f(torch.randn(B, *self.latent_shape[1:], device=device))

        T = forward_steps
        if sampling_loop_angle is None:
            sampling_loop_angle = init_angle
        angles = [min(self.max_angle_deg, sampling_loop_angle)] * max(T - 1, 0)

        outs = []
        for t in torch.linspace(0, 1, num_interp).tolist():
            e_t = e_a if fix_latent else self.f(slerp(e_a, e_b, t, slerp_dim))
            y_t, c_t = cond_at(t)
            self.cached_noise = e_t if cache_sampling_noise else None

            alpha = self._dec_angle(B, min(self.max_angle_deg, init_angle), device)
            x = self.decoder(e_t, y_t, alpha=alpha, cond_embed=c_t)
            for angle_deg in angles:
                z = self.encoder(x, y_t)
                z, alpha = self.spherify(
                    z,
                    angle_deg,
                    cache_noise=cache_sampling_noise,
                )
                x = self.decoder(z, y_t, alpha=alpha, cond_embed=c_t)
            outs.append(torch.clamp(x * 0.5 + 0.5, 0, 1))
        return outs

    @torch.no_grad()
    def reconstruct_loop(
        self,
        x,
        y=None,
        rec_angle=85.0,
        forward_steps=1,
        sampling_loop_angle=None,
        cache_sampling_noise=True,
        return_step_images=False,
    ):
        """
        reconstruction followed by the multi-step sampling loop of `generate`.
        step 0 encodes x and decodes its clean latent while telling the decoder
        the angle is rec_angle, so no noise is added but the decoder is free to
        move toward class y. steps 1..T-1 are the usual encode -> spherify ->
        decode loop, where the input's content fades step by step

        x                    : [B, 3, H, W] in [-1, 1]
        y                    : [B] class ids, None for the null class
        rec_angle            : decoder angle (deg) of step 0; 0 is the plain round trip
        forward_steps        : total decode steps T
        sampling_loop_angle  : angle (deg) of steps 1..T-1; None for rec_angle
        cache_sampling_noise : reuse one noise for every step of the loop
        return_step_images   : also return the list of per-step images
        out                  : [B, 3, H, W] in [0, 1]
        """
        self.cached_noise = None  # clean the cache

        B = x.shape[0]
        if self.num_classes > 0 and y is None:
            y = torch.full((B,), self.num_classes, device=x.device, dtype=torch.long)
        elif self.num_classes == 0:
            y = None

        # step 0: decode the clean latent at rec_angle
        z = self.f(self.encoder(x, y))
        alpha = self._dec_angle(B, min(self.max_angle_deg, rec_angle), x.device)
        x = self.decoder(z, y, alpha=alpha)

        h = torch.clamp(x * 0.5 + 0.5, 0, 1)
        if forward_steps <= 1:
            return (h, [h]) if return_step_images else h
        step_images = [h]

        # steps 1..T-1: same refinement loop as generate
        T = forward_steps
        if sampling_loop_angle is None:
            sampling_loop_angle = rec_angle
        angles = [min(self.max_angle_deg, sampling_loop_angle)] * (T - 1)
        logger.info(
            f"[reconstruct_loop] rec_angle={rec_angle:.2f}deg, loop angles: "
            f"{[round(a, 4) for a in angles]} deg"
        )

        for angle_deg in angles:
            z = self.encoder(x, y)
            z, alpha = self.spherify(
                z,
                angle_deg,
                cache_noise=cache_sampling_noise,
            )
            x = self.decoder(z, y, alpha=alpha)
            if return_step_images:
                step_images.append(torch.clamp(x * 0.5 + 0.5, 0, 1))

        x = torch.clamp(x * 0.5 + 0.5, 0, 1)
        return (x, step_images) if return_step_images else x
