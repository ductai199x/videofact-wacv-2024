import importlib
from .videofact_pl_wrapper import VideoFACTPLWrapper

default_config = {
    "img_size": (1080, 1920),
    "in_chans": 3,
    "patch_size": 128,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4,
    "drop_rate": 0,
    "bb1_db_depth": 1,
    "loss_alpha": 0.4,
    "lr": 1.0e-04,
    "decay_step": 2,
    "decay_rate": 0.75,
    "fe": "mislnet",
    "fe_config": {
        "patch_size": 128,
        "num_classes": 33,
    },
    "fe_ckpt": "/home/tai/df_models/lab04_220401_mislnet_video_v1_ep=57_vl=0.6216.ckpt",
    "fe_freeze": False,
}

available_ablations_codenames = [
    "1_templates",  # 0
    "10_templates",  # 1
    "concat_scaled_fe",  # 2
    "no_cfe",  # 3
    "no_pfe",  # 4
    "no_scaling_no_combine_fe_embed",  # 5
    "no_transformer",  # 6
    "no_transformer_yes_templates",  # 7
    "conv1x1",  # 8
    "conv2d_classifier",  # 9
]


def prepare_model(ablation_codename: str, prev_ckpt: str, config: dict = None):
    if ablation_codename == "":
        import_path = f"src.model.common.videofact"
    else:
        if ablation_codename not in available_ablations_codenames:
            raise NotImplementedError(
                f"{ablation_codename} does not exists. Available codenames for v11 are: {available_ablations_codenames}"
            )
        import_path = f"src.model.ablations.videofact_{ablation_codename}"
    torch_model = importlib.import_module(import_path).VideoFACT

    print("\n" + "#" * 40 + "\n" + f"LOADING MODEL: {import_path}" + "\n" + "#" * 40 + "\n")
    if config is None:
        config = default_config
    if prev_ckpt:
        print(f"Loading from checkpoint: {prev_ckpt}...")
        return VideoFACTPLWrapper.load_from_checkpoint(prev_ckpt, model=torch_model, **config)
    else:
        return VideoFACTPLWrapper(model=torch_model, **config)
