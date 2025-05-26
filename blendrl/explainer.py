from datetime import datetime
from typing import Union

import numpy as np
import torch as th
import pygame
import vidmaker

from nudge.agents.logic_agent import NsfrActorCritic
from nudge.agents.neural_agent import ActorCritic
from nudge.utils import load_model, yellow
from nudge.env import NudgeBaseEnv
from captum.attr import visualization as viz

SCREENSHOTS_BASE_PATH = "out/screenshots/"
PREDICATE_PROBS_COL_WIDTH = 500 * 2
FACT_PROBS_COL_WIDTH = 1000
CELL_BACKGROUND_DEFAULT = np.array([40, 40, 40])
CELL_BACKGROUND_HIGHLIGHT = np.array([40, 150, 255])
CELL_BACKGROUND_HIGHLIGHT_POLICY = np.array([234, 145, 152])
CELL_BACKGROUND_SELECTED = np.array([80, 80, 80])


class Explainer:
    model: Union[NsfrActorCritic, ActorCritic]
    window: pygame.Surface
    clock: pygame.time.Clock

    def __init__(self,
                 agent_path: str = None,
                 env_name: str = "seaquest",
                 device: str = "cpu",
                 fps: int = None,
                 deterministic=True,
                 env_kwargs: dict = None,
                 render_predicate_probs=True):

        self.fps = fps
        self.deterministic = deterministic
        self.render_predicate_probs = render_predicate_probs

        # Load model and environment
        self.model = load_model(agent_path, env_kwargs_override=env_kwargs, device=device, explain=True)
        self.env = NudgeBaseEnv.from_name(env_name, mode='deictic', seed=10, **env_kwargs)
        # self.env = self.model.env
        self.env.reset()
        
        print(self.model._print())

        print(f"Playing '{self.model.env.name}' with {'' if deterministic else 'non-'}deterministic policy.")

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

        self.predicates = self.model.logic_actor.prednames

        self._init_pygame()

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.reset = False
        self.takeover = False
        

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Environment")
        frame = self.env.env.render()
        self.env_render_shape = frame.shape[:2]
        window_shape = list(self.env_render_shape)
        if self.render_predicate_probs:
            window_shape[0] += PREDICATE_PROBS_COL_WIDTH
        self.window = pygame.display.set_mode(window_shape, pygame.SCALED)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Calibri', 24)

    def run(self):
        length = 0
        ret = 0

        obs, obs_nn = self.env.reset()
        obs_nn = th.tensor(obs_nn, device=self.model.device) 
        # print(obs_nn.shape)

        counter = 0
        while self.running:
            counter += 1
            self.reset = False
            self._handle_user_input()
            if not self.paused:
                if not self.running:
                    break  # outer game loop

                if self.takeover:  # human plays game manually
                    # assert False, "Unimplemented."
                    action = self._get_action()
                else:  # AI plays the game
                    # print("obs_nn: ", obs_nn.shape)
                    action, prob = self.model.act(obs_nn, obs)  # update the model's internals
                    value = self.model.get_value(obs_nn, obs)
                    
                    # Get Explanation
                    # prob.backward(retain_graph=True)
                    # atom_grads = NAEUMANN.mpm.dummy_zeros.grad.squeeze(-1).unsqueeze(0)
                    if counter % 5 == 0:
                        neural_explanation, logic_explanation, weights = self.model.actor.get_explanation(obs_nn, obs, action)
                    
                        # original_input_size = (self.env.raw_state_ori.shape[0], self.env.raw_state_ori.shape[1])
                        frame = self.get_render_frame()
                        original_input_size = frame.shape[:2]
                        logic_explanation_image = self.generate_logic_attribution_map(self.model.actor.logic_actor.atoms, obs, logic_explanation, size=original_input_size)
                        visualize_attributions(action, frame, obs, neural_explanation, logic_explanation_image, weights, counter, self.env.name)
                        print("Action: ", action)
                    

                (new_obs, new_obs_nn), reward, done, terminations, infos = self.env.step(action, is_mapped=self.takeover)
                if reward > 0:
                    print(f"Reward: {reward:.2f}")
                new_obs_nn = th.tensor(new_obs_nn, device=self.model.device) 
                

                self._render()

                if self.takeover and float(reward) != 0:
                    print(f"Reward {reward:.2f}")

                if self.reset:
                    done = True
                    new_obs = self.env.reset()
                    self._render()

                obs = new_obs
                obs_nn = new_obs_nn
                length += 1

                if done:
                    print(f"Return: {ret} - Length {length}")
                    ret = 0
                    length = 0
                    self.env.reset()

        pygame.quit()

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

    def _handle_user_input(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_r:  # 'R': reset
                    self.reset = True

                elif event.key == pygame.K_f:  # 'F': fast forward
                    self.fast_forward = not(self.fast_forward)

                elif event.key == pygame.K_t:  # 'T': trigger takeover
                    if self.takeover:
                        print("AI takeover")
                    else:
                        print("Human takeover")
                    self.takeover = not self.takeover
                
                elif event.key == pygame.K_o:  # 'O': toggle overlay
                    self.env.env.render_oc_overlay = not(self.env.env.render_oc_overlay)

                elif event.key == pygame.K_c:  # 'C': capture screenshot
                    file_name = f"{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}.png"
                    pygame.image.save(self.window, SCREENSHOTS_BASE_PATH + file_name)

                elif (event.key,) in self.keys2actions.keys():  # env action
                    self.current_keys_down.add(event.key)

            elif event.type == pygame.KEYUP:  # keyboard key released
                if (event.key,) in self.keys2actions.keys():
                    self.current_keys_down.remove(event.key)

                # elif event.key == pygame.K_f:  # 'F': fast forward
                #     self.fast_forward = False

    def _render(self):
        self.window.fill((20, 20, 20))  # clear the entire window
        self._render_policy_probs()
        self._render_predicate_probs()
        self._render_neural_probs()
        self._render_env()

        pygame.display.flip()
        pygame.event.pump()
        if not self.fast_forward:
            self.clock.tick(self.fps)

    def _render_env(self):
        frame = self.env.env.render()
        frame_surface = pygame.Surface(self.env_render_shape)
        pygame.pixelcopy.array_to_surface(frame_surface, frame)
        self.window.blit(frame_surface, (0, 0))
        
    def get_render_frame(self):
        frame = self.env.env.render()
        # frame_surface = pygame.Surface(self.env_render_shape)
        # pygame.pixelcopy.array_to_surface(frame_surface, frame)
        return frame

    def _render_policy_probs_rows(self):
        anchor = (self.env_render_shape[0] + 10, 25)

        model = self.model
        policy_names = ['neural', 'logic']
        weights = model.get_policy_weights()
        for i, w_i in enumerate(weights):
            w_i = w_i.item()
            name = policy_names[i]
            # Render cell background
            color = w_i * CELL_BACKGROUND_HIGHLIGHT_POLICY + (1 - w_i) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                PREDICATE_PROBS_COL_WIDTH - 12,
                28
            ])
            # print(w_i, name)

            text = self.font.render(str(f"{w_i:.3f} - {name}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)
            
    def _render_policy_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)

        model = self.model
        policy_names = ['neural', 'logic']
        weights = model.get_policy_weights()
        for i, w_i in enumerate(weights):
            w_i = w_i.item()
            name = policy_names[i]
            # Render cell background
            color = w_i * CELL_BACKGROUND_HIGHLIGHT_POLICY + (1 - w_i) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2 + i * 500,
                anchor[1] - 2,
                (PREDICATE_PROBS_COL_WIDTH / 2 - 12) * w_i,
                28
            ])

            text = self.font.render(str(f"{w_i:.3f} - {name}"), True, "white", None)
            text_rect = text.get_rect()
            if i == 0:
                text_rect.topleft = (self.env_render_shape[0] + 10, 25) 
            else:
                text_rect.topleft = (self.env_render_shape[0] + 10 + i * 500, 25)
            self.window.blit(text, text_rect)
        
    def _render_predicate_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)
        nsfr = self.model.actor.logic_actor
        pred_vals = {pred: nsfr.get_predicate_valuation(pred, initial_valuation=False) for pred in nsfr.prednames}
        for i, (pred, val) in enumerate(pred_vals.items()):
            i += 2
            # Render cell background
            color = val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2 + PREDICATE_PROBS_COL_WIDTH / 2,
                anchor[1] - 2 + i * 35,
                (PREDICATE_PROBS_COL_WIDTH /2  - 12) * val,
                28
            ])

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10 + PREDICATE_PROBS_COL_WIDTH / 2, 25 + i * 35)
            self.window.blit(text, text_rect)
            
            
    def _render_neural_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)
        blender_actor = self.model.actor
        action_vals = blender_actor.neural_action_probs[0].detach().cpu().numpy()
        action_names = ["noop", "fire", "up", "right", "left", "down", "upright", "upleft", "downright", "downleft", "upfire", "rightfire", "leftfire", "downfire", "uprightfire", "upleftfire", "downrightfire", "downleftfire"]
        for i, (pred, val) in enumerate(zip(action_names, action_vals)):
            i += 2
            # Render cell background
            color = val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                (PREDICATE_PROBS_COL_WIDTH / 2  - 12) * val,
                28
            ])

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)

            
    def _render_facts(self, th=0.1):
        anchor = (self.env_render_shape[0] + 10, 25)

        # nsfr = self.nsfr_reasoner
        nsfr = self.model.actor.logic_actor
        
        fact_vals = {}
        v_T = nsfr.V_T[0]
        preds_to_skip = ['.', 'true_predicate', 'test_predicate_global', 'test_predicate_object']
        for i, atom in enumerate(nsfr.atoms):
            if v_T[i] > th:
                if atom.pred.name not in preds_to_skip:
                    fact_vals[atom] = v_T[i].item()
                
        for i, (fact, val) in enumerate(fact_vals.items()):
            i += 2
            # Render cell background
            color = val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                FACT_PROBS_COL_WIDTH - 12,
                28
            ])

            text = self.font.render(str(f"{val:.3f} - {fact}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)


    import matplotlib.pyplot as plt

    def generate_logic_attribution_map(self, atoms, logic_state, logic_explanation, size=(84, 84)):
        """Generates a heatmap of the logic attribution map and returns an array."""
        logic_state = logic_state[-1] # get the last example in the batch
        # neumann has transposed attributoin vector
        if logic_explanation.size(0) != 1:
            attributions = torch.transpose(logic_explanation, 1, 0)[0]
        else:
            attributions = logic_explanation[-1] # get the last example in
        object_value_dic = {}
        for i, atom in enumerate(atoms):
            if attributions[i].item() < 0.9:
                continue
            for term in atom.terms:
                if "obj" in str(term):
                    if term not in object_value_dic:
                        object_value_dic[term] = attributions[i].item()
                    else:
                        object_value_dic[term] = max(object_value_dic[term], attributions[i].item())



        xy_list = []
        bboxes = []
        values = []
        for term, value in object_value_dic.items():
            # obj1 -> 0, obj2 -> 1, ...
            object_id = int(str(term).split("obj")[-1]) - 1
            # xy_coord = logic_state[object_id][1:3]
            bbox = self.env.bboxes[object_id]
            bboxes.append(bbox)
            values.append(value)
            # xy_value_dic[xy_coord] = value
        # generate attribution by Gaussian
        Y, X, _ = self.env.env.get_rgb_state.shape
        # upscale factor: 4
        size = (X * 4, Y * 4)
        attribution_map = np.zeros(size) 
        for bbox, value in zip(bboxes, values):
            x, y, w, h = bbox * 4
            attribution_map[x:x+w, y:y+h] +=  value
        return attribution_map.transpose(1, 0)
    
        
def make_gaussian(size=50):
    # Importing the NumPy library and aliasing it as 'np'
    # Generating 2D grids 'x' and 'y' using meshgrid with 10 evenly spaced points from -1 to 1
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))

    # Calculating the Euclidean distance 'd' from the origin using the generated grids 'x' and 'y'
    d = np.sqrt(x*x + y*y)

    # Defining parameters sigma and mu for a Gaussian-like distribution
    sigma, mu = 1.0, 0.0

    # Calculating the Gaussian-like distribution 'g' based on the distance 'd', sigma, and mu
    g = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))

    # Printing a message indicating a 2D Gaussian-like array will be displayed
    print("2D Gaussian-like array:")

    # Printing the calculated 2D Gaussian-like array 'g'
    print(g)
    return g

        
    
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
def visualize_attributions(action, neural_state, logic_state, neural_explanation, logic_explanation, weights, step, env_name):
    plt.set_cmap('cividis')
    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off
    
    # plt.tick_params(
    #     axis='y',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    size = (neural_state.shape[0], neural_state.shape[1])
    # neural_state = torch.Tensor(neural_state)#
    # frame = env.env.render()
    # frame_surface = pygame.Surface(env, env.env_render_shape)
    # axes[0, 0].imshow(neural_state[:, -1, :, :].squeeze().cpu().detach().numpy())
    axes[0, 0].imshow(neural_state.transpose(1, 0, 2 ))
    axes[0, 0].set_title("Neural State")
    neural_explanation = neural_explanation[:, -1, :, :].squeeze().cpu().detach().numpy()
    neural_explanation = Image.fromarray(neural_explanation) #.resize(size)
    neural_explanation = np.asarray(neural_explanation.resize(size))
    # neural_explanation = neural_explanation.transpose(1, 0)
    # neural_explanation = np.resize(neural_explanation, (neural_state.shape[0], neural_state.shape[1]))
    blended_explanation = neural_explanation * weights[0] + logic_explanation * weights[1]
    
    blended_explanation_image = axes[1, 1].imshow(blended_explanation)
    axes[1, 1].set_title("Blended Explanation")
    blended_explanation_image.set_clim(0, 1)
    # fig.colorbar(blended_explanation_image, ax=axes[0, 1])
    neural_explanation_image = axes[1, 0].imshow(neural_explanation)
    axes[1, 0].set_title("Neural Explanation")
    neural_explanation_image.set_clim(0, 1)
    # axes[1, 0].set_colorbar()
    # fig.colorbar(neural_explanation_image, ax=axes[1, 0])
    logic_explanation_image = axes[1, 2].imshow(logic_explanation)
    axes[1, 2].set_title("Logic Explanation")
    logic_explanation_image.set_clim(0, 1)
    # fig.colorbar(logic_explanation_image, ax=axes[1, 1])
    # axes[1, 1].set_colorbar()
    # axes.xaxis.set_ticklabels([])
    # axes.yaxis.set_ticklabels([])

    # binary_input_state = Image.fromarray(neural_state.transpose(1, 0, 2 )).convert(1)
    # masked_input_state = binary_input_state * blended_explanation
    heatmap_img = cv2.applyColorMap(np.uint8(255 * blended_explanation), cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    # heatmap_img = blended_explanation
    # hetamap_plot_image = 
    # heatmap_img = cv2.applyColorMap(np.uint8(255 * blended_explanation), cv2.COLORMAP_JET)
    # fig.colorbar(hetamap_plot_image, ax=axes[0, 2])
    # heatmap_img = blended_explanation
    # img = cv2.applyColorMap(neural_state.transpose(1, 0, 2), cv2.COLORMAP_GRA)#Image.fromarray(neural_state.transpose(1, 0, 2 )).convert('RGB')
    img = cv2.cvtColor(neural_state.transpose(1, 0, 2), cv2.COLOR_RGB2GRAY)#.astype(np.float64)
    img = torch.tensor(img).unsqueeze(-1).expand(-1, -1, 3).numpy()
    # img = neural_state.transpose(1, 0, 2)
    maked_input_state = cv2.addWeighted(heatmap_img, 0.6, img, 0.4, 0)

    # masked_input_state = np.where(blended_explanation > 0.5, neural_state.transpose(1, 0 , 2), np.array(255, 255, 255))
    masked_image = axes[0, 1].imshow(maked_input_state)
    axes[0, 1].set_title("Masked State")
    # masked_image.set_clim(0, 1)
    # fig.colorbar(masked_image, ax=axes[0, 2])
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    axes[0, 2].axis('off')
    
    # Set the colorbar maximum and minimum values
    folder = f"out/explanations/{env_name}/"

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{step}.png")
    
    plt.savefig(path)
    plt.close()

    # plt.show()
    
    