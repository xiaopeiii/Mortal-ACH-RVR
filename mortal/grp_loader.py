import logging
from typing import Dict, Tuple

import torch

from model import GRP


def _infer_grp_network_from_state(grp_state: Dict) -> Dict[str, int]:
    model_state = grp_state.get("model", grp_state)
    weight_ih = model_state.get("rnn.weight_ih_l0")
    if weight_ih is None:
        return {"hidden_size": 64, "num_layers": 2}

    hidden_size = int(weight_ih.shape[0] // 3)
    num_layers = sum(1 for k in model_state.keys() if k.startswith("rnn.weight_ih_l"))
    if num_layers <= 0:
        num_layers = 2

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }


def load_grp_from_cfg(grp_cfg: Dict, *, map_location=torch.device("cpu")) -> Tuple[GRP, Dict[str, int]]:
    grp_state = torch.load(
        grp_cfg["state_file"],
        weights_only=True,
        map_location=map_location,
    )
    network_cfg = grp_cfg.get("network")
    if network_cfg is None:
        network_cfg = _infer_grp_network_from_state(grp_state)
        logging.info(
            "grp.network is missing in config, inferred from checkpoint: hidden_size=%s, num_layers=%s",
            network_cfg["hidden_size"],
            network_cfg["num_layers"],
        )

    grp = GRP(**network_cfg)
    grp.load_state_dict(grp_state["model"])
    return grp, network_cfg
