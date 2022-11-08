import torch
import torch.nn as nn

# RL environment
import gym
import minerl

# Visualization
from colabgymrender.recorder import Recorder
from pyvirtualdisplay import Display
from IPython.display import HTML

# Others
import numpy as np
from tqdm.notebook import tqdm
import logging

logging.disable(logging.ERROR)

# Create virtual display
display = Display(visible=0, size=(400, 300))
display.start()


class CNN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations):
        return self.cnn(observations)


def dataset_action_batch_to_actions(dataset_actions, camera_margin=5):
    camera_actions = dataset_actions["camera"].squeeze()
    attack_actions = dataset_actions["attack"].squeeze()
    forward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    batch_size = len(camera_actions)
    actions = np.zeros((batch_size,), dtype=int)

    for i in range(len(camera_actions)):
        if camera_actions[i][0] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        else:
            actions[i] = -1
    return actions


class ActionShaping(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10):
        super().__init__(env)
        self.camera_angle = camera_angle
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            [('jump', 1)],
            [('camera', [-self.camera_angle, 0])],
            [('camera', [self.camera_angle, 0])],
            [('camera', [0, self.camera_angle])],
            [('camera', [0, -self.camera_angle])],
        ]
        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
                act['attack'] = 1
            self.actions.append(act)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


# Get data
minerl.data.download(directory='data', environment='MineRLTreechop-v0')
data = minerl.data.make("MineRLTreechop-v0", data_dir='data', num_workers=2)

# Model
model = CNN((3, 64, 64), 7).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training loop
step = 0
losses = []
for state, action, _, _, _ \
        in tqdm(data.batch_iter(num_epochs=6, batch_size=32, seq_len=1)):
    # Get pov observations
    obs = state['pov'].squeeze().astype(np.float32)
    # Transpose and normalize
    obs = obs.transpose(0, 3, 1, 2) / 255.0

    # Translate batch of actions for the ActionShaping wrapper
    actions = dataset_action_batch_to_actions(action)

    # Remove samples with no corresponding action
    mask = actions != -1
    obs = obs[mask]
    actions = actions[mask]

    # Update weights with backprop
    logits = model(torch.from_numpy(obs).float().cuda())
    loss = criterion(logits, torch.from_numpy(actions).long().cuda())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss
    step += 1
    losses.append(loss.item())
    if (step % 2000) == 0:
        mean_loss = sum(losses) / len(losses)
        tqdm.write(f'Step {step:>5} | Training loss = {mean_loss:.3f}')
        losses.clear()

torch.save(model.state_dict(), 'model.pth')
del data

model = CNN((3, 64, 64), 7).cuda()
model.load_state_dict(torch.load('model.pth'))

env = gym.make('MineRLObtainDiamond-v0')
env1 = Recorder(env, './video', fps=60)
env = ActionShaping(env1)

action_list = np.arange(env.action_space.n)

obs = env.reset()

for step in tqdm(range(1000)):
    # Get input in the correct format
    obs = torch.from_numpy(obs['pov'].transpose(2, 0, 1)[None].astype(np.float32) / 255).cuda()
    # Turn logits into probabilities
    probabilities = torch.softmax(model(obs), dim=1)[0].detach().cpu().numpy()
    # Sample action according to the probabilities
    action = np.random.choice(action_list, p=probabilities)

    obs, reward, _, _ = env.step(action)

env1.play()
