import os
import torch

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config=None, version=None, device=None, is_half=None):
    if config is not None:
        device = config.device
        is_half = config.is_half
    if version != "v2_mod":
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["assets/hubert/hubert_base.pt"],
            suffix="",
        )
        hubert_model = models[0]
        hubert_model = hubert_model.to(device)
        if is_half:
            hubert_model = hubert_model.half()
        else:
            hubert_model = hubert_model.float()
    else:
        from transformers.models.hubert.modeling_hubert import HubertModel

        hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        hubert_model.load_state_dict(
            torch.load("assets/custom/speaker_disentangled_hubert.pt")
        )
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.float()
    return hubert_model.eval()
