import os
import time
import torch
from model import Brain, DQN
from config import config


def main() -> None:
    version = config["control"]["version"]
    mortal = Brain(version=version, **config["resnet"]).eval()
    dqn = DQN(version=version).eval()

    state = {
        "mortal": mortal.state_dict(),
        "current_dqn": dqn.state_dict(),
        "config": config,
        "timestamp": time.time(),
    }

    targets = {
        config["baseline"]["train"]["state_file"],
        config["baseline"]["test"]["state_file"],
    }
    for filename in targets:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        print(f"saved: {filename}")


if __name__ == "__main__":
    main()
