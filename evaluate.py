import tyro
from nudge.evaluator import Evaluator

from dataclasses import dataclass
import tyro


def main(
    env_name: str = "seaquest",
    agent_path: str = "out/runs/models/kangaroo_demo",
    fps: int = 5,
    episodes: int = 2,
    model: str = "blendrl",
    device: str = "cuda:0",
) -> None:
    """
    Ablation evaluation script. This script evaluates the performance of the blendrl agent with different actor by tracking
    the performance of the agent with different actor modes (hybrid, logic, neural) and different blending modes (logic, neural).
    """
    evaluator = Evaluator(
        episodes=episodes,
        agent_path=agent_path,
        env_name=env_name,
        fps=fps,
        deterministic=False,
        device=device,
        # env_kwargs=dict(render_oc_overlay=True),
        env_kwargs=dict(render_oc_overlay=False),
        render_predicate_probs=True,
    )
    evaluator.run()


if __name__ == "__main__":
    tyro.cli(main)
