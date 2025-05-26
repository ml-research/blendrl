from datetime import datetime
from typing import Union

import numpy as np
import torch as th
# import pygame
# import vidmaker

from nudge.agents.logic_agent import NsfrActorCritic
from nudge.agents.neural_agent import ActorCritic
from nudge.utils import load_neuralppo_model, yellow
from nudge.env import NudgeBaseEnv

SCREENSHOTS_BASE_PATH = "out/screenshots/"
PREDICATE_PROBS_COL_WIDTH = 500 * 2
FACT_PROBS_COL_WIDTH = 1000
CELL_BACKGROUND_DEFAULT = np.array([40, 40, 40])
CELL_BACKGROUND_HIGHLIGHT = np.array([40, 150, 255])
CELL_BACKGROUND_HIGHLIGHT_POLICY = np.array([234, 145, 152])
CELL_BACKGROUND_SELECTED = np.array([80, 80, 80])

import torch
torch.set_num_threads(5)

class EvaluatorNeuralPPO:
    model: Union[NsfrActorCritic, ActorCritic]
    # window: pygame.Surface
    # clock: pygame.time.Clock

    def __init__(self,
                 agent_path: str = None,
                 env_name: str = "seaquest",
                 device: str = "cpu",
                 fps: int = None,
                 deterministic=True,
                 env_kwargs: dict = None,
                 render_predicate_probs=True,
                 episodes: int = 2,
                 seed=0):

        self.fps = fps
        self.deterministic = deterministic
        self.render_predicate_probs = render_predicate_probs
        self.episodes = episodes
        self.agent_path = agent_path
        self.env_name = env_name
        
        self.device = torch.device(device)

        # Load model and environment

        self.model = load_neuralppo_model(agent_path, env_kwargs_override=env_kwargs, device=device)
        self.env = NudgeBaseEnv.from_name(env_name, mode='deictic', seed=seed, **env_kwargs)
        # self.env = self.model.env
        self.env.reset()
        
        # print(self.model._print())

        # print(f"Playing '{self.model.env.name}' with {'' if deterministic else 'non-'}deterministic policy.")

        if fps is None:
            fps = 15
        self.fps = fps

        try:
            self.action_meanings = self.env.env.get_action_meanings()
            self.keys2actions = self.env.env.unwrapped.get_keys_to_action()
        except Exception:
            print(yellow("Info: No key-to-action mapping found for this env. No manual user control possible."))
            self.action_meanings = None
            self.keys2actions = {}
        self.current_keys_down = set()

        # self.predicates = self.model.logic_actor.prednames

        # self._init_pygame()

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.reset = False
        self.takeover = False
        

    # def _init_pygame(self):
    #     pygame.init()
    #     pygame.display.set_caption("Environment")
    #     frame = self.env.env.render()
    #     self.env_render_shape = frame.shape[:2]
    #     window_shape = list(self.env_render_shape)
    #     if self.render_predicate_probs:
    #         window_shape[0] += PREDICATE_PROBS_COL_WIDTH
    #     self.window = pygame.display.set_mode(window_shape, pygame.SCALED)
    #     self.clock = pygame.time.Clock()
    #     self.font = pygame.font.SysFont('Calibri', 24)

    def run(self):
        length = 0
        ret = 0

        obs, obs_nn = self.env.reset()
        obs_nn = th.tensor(obs_nn, device=self.device) 
        obs = obs.to(self.device)
        # print(obs_nn.shape)

        episode_count = 0
        returns = []
        lengths =[]
        blend_entropies = []
        while self.running:
            self.reset = False
            # self._handle_user_input()
            if not self.paused:
                if not self.running:
                    break  # outer game loop

                if self.takeover:  # human plays game manually
                    # assert False, "Unimplemented."
                    action = self._get_action()
                else:  # AI plays the game
                    # print("obs_nn: ", obs_nn.shape)
                    # get blend entropy
                    # _, newlogprob, entropy, blend_entropy, newvalue = self.model.get_action_and_value(obs_nn, obs, action)
                    action, logprob, _, value = self.model.get_action_and_value(obs_nn)
                    # blend_entropies.append(blend_entropy.detach().item())

                (new_obs, new_obs_nn), reward, done, terminations, infos = self.env.step(action, is_mapped=self.takeover)
                if reward > 0:
                    print(f"Reward: {reward:.2f}")
                new_obs_nn = th.tensor(new_obs_nn, device=self.device) 
                

                # self._render()

                if self.takeover and float(reward) != 0:
                    print(f"Reward {reward:.2f}")

                if self.reset:
                    done = True
                    new_obs = self.env.reset()
                    # self._render()

                obs = new_obs
                obs = obs.to(self.device)
                obs_nn = new_obs_nn
                obs_nn = obs_nn.to(self.device)  
                length += 1

                if terminations:
                    if "final_info" in infos: # or next_done.any():
                        info = infos['final_info']
                        # final_info = info['final_info']
                        if "episode" in info:
                            # print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                            # print("Return: ", info["episode"]["r"], "Length: ", info["episode"]["l"])
                            # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            ret = info["episode"]["r"][0]
                            length = info["episode"]["l"][0]
                            print(f"Return: {ret} - Length {length}")
                            episode_count += 1
                            returns.append(ret)
                            lengths.append(length)
                    self.env.reset()
                
                if episode_count >= self.episodes:
                    break
                
        # compute mean and std
        mean_returns = np.mean(np.array(returns))
        std_returns = np.std(np.array(returns))
        mean_lengths = np.mean(np.array(lengths))
        std_lengths = np.std(np.array(lengths))
        mean_bents = np.round(np.mean(np.array(blend_entropies)), 3)
        std_bents = np.round(np.std(np.array(blend_entropies)), 3)
        # save the returns and lengths to a pickle file
        import pickle
        if 'blender_neural' in self.agent_path:
            blender_mode = 'neural'
        else:
            blender_mode = 'logic'
        with open(f'out/eval/{self.env_name}_neuralppo_returns.pkl', 'wb') as f:
            print(f"Returns: {mean_returns} ± {std_returns}")
            pickle.dump((mean_returns, std_returns), f)
        with open(f'out/eval/{self.env_name}_neuralppo_lengths.pkl', 'wb') as f:
            print(f"Lengths: {mean_lengths} ± {std_lengths}")
            pickle.dump((mean_lengths, std_lengths), f)
        with open(f'out/eval/{self.env_name}_neuralppo_blend_entropies.pkl', 'wb') as f:
            print(f"Blend Entropies: {mean_bents} ± {std_bents}")
            pickle.dump((mean_bents, std_bents), f)
        

        # pygame.quit()

    def _get_action(self):
        if self.keys2actions is None:
            return 0  # NOOP
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

