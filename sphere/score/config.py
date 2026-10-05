config = {
    "score": {
        # score net: an edm-preconditioned token denoiser
        "arch": {
            "width": 384,
            "width_margin": 0,  # extra width on top of the feature dim
            "depth": 2,
            "real_depth": None,  # None -> depth
            "fake_depth": None,  # None -> depth
            "num_heads": 8,
            "drop_residual_path_prob": 0.1,
            "use_qk_norm": True,
            "sigma_data": 0.5,
            "sdpa_mode": "sdpa",
            "adaln_single": False,
            "sigma_cond": True,
            "class_cond": False,
        },
        # optimizer of the real and fake score nets
        "optimizer": {
            "lr": 1.0e-4,
            "min_lr": 1.0e-5,
            "real_min_lr": None,  # None -> min_lr
            "fake_min_lr": None,  # None -> min_lr
            "warmup_epochs": 0,
            "decay_epochs": 0,  # 0 -> constant lr
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,  # per score net; 0 -> off
        },
        # noise levels: plain keys train the score nets, gen_* drive the generator
        "sigma": {
            "p_mean": -1.2,
            "p_std": 1.2,
            "min": 2.0e-3,
            "max": 8.0,
            "gen_p_mean": None,  # None -> p_mean
            "gen_p_std": None,  # None -> p_std
            "gen_max": None,  # None -> max
        },
        # augmentation of the images fed to the featurizer
        "augment": {
            "enabled": False,
            "type": "diffaug",  # diffaug | flip
            "symmetric": False,  # also augment the real images
            "translation_prob": 0.0,
            "flip_prob": 0.0,
            "translation_ratio": 0.125,
            "translation_padding": "reflect",  # zeros | reflect | circular
        },
        # frozen featurizer, used when loss.space = dino
        "dino": {
            "ckpt_path": "workspace/pretrained/discs/dino_vit_small_patch8_224.pth",
            "recipe": "S_8",  # S_16 | S_8 | B_16
            "layers": [5, 11],
            "img_size": 256,
        },
        # frozen featurizer, used when loss.space = convnext
        "convnext": {
            "model_name": "timm/convnextv2_nano.fcmae_ft_in1k",
            "stages": [1, 2],  # 0-3: stride 4/8/16/32, dims 96/192/384/768
            "img_size": 256,  # images are resized to this
        },
    },
    "loss": {
        "space": "convnext",  # dino | convnext
        "weight": 1.0,
        "compile": False,  # torch.compile the score nets
        "apply_regime": "hi",  # all | hi | lo
        "grad_clip": 0.0,  # per-sample clip of the generator gradient; 0 -> off
        "skip_load_optim_state": False,  # on resume, restart the score optimizer
        "updates": 1,  # score-net steps per training iteration
        "sync_interval": 1,  # sync the score nets across ranks every n updates
        # schedule, in epochs
        "start_epoch": 0,
        "real_start_epoch": None,  # None -> start_epoch
        "fake_start_epoch": None,  # None -> start_epoch
        "gen_start_epoch": None,  # None -> start_epoch
        "gen_warmup_start_epoch": None,  # None -> gen_start_epoch - 10
        "gen_warmup_shape": "cosine",  # linear | cosine
        # feature standardization
        "standardize": False,
        "standardize_per_channel": False,
        "standardize_channel_floor": 0.1,
        "scale_calib_steps": None,  # None -> 1000
        "scale_ema_decay": 0.99,  # used after calibration
        # normalizer of the generator gradient
        "normalizer_reduction": "batch",  # sample | batch
        "normalizer_floor": 1e-6,
        # balance the gradient magnitude across the scored layers
        "transport_balance": False,
        "transport_balance_target": None,  # None -> set during calibration
        "transport_balance_decay": 0.99,
        "transport_balance_calib_steps": 1000,
        # extra term: semantic alignment loss
        "logits_weight": 0.0,
        "logits_loss_type": "cosine",  # mse | cosine
        "logits_start_epoch": None,  # None -> start_epoch
        # extra term: latent consistency loss
        "latent_weight": 0.0,
        "latent_loss_type": "cosine",  # mse | cosine
        "latent_level": "pooled",  # pooled | tokens
        "latent_start_epoch": None,  # None -> start_epoch
        # extra term: frechet distance on the final features (sphere/fd_loss.py)
        "fd_weight": 0.0,
        "fd_start_epoch": None,  # None -> start_epoch
        "fd_pool": "mean",  # mean | token
        "fd_ema_decay": 0.999,
        "fd_use_clean_ema": True,
        "fd_use_noisy_ema": False,
        "fd_norm_eps": 0.01,
    },
}
