import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

import Utilities


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# meta controller network
class hDQN(nn.Module):
    def __init__(self, params):
        # utilities = Utilities.Utilities()
        self.params = params
        super(hDQN, self).__init__()
        env_layer_num = self.params.OBJECT_TYPE_NUM + 1  # +1 for agent layer

        kernel_size = 2
        self.conv1 = nn.Conv2d(in_channels=env_layer_num,
                               out_channels=self.params.DQN_CONV1_OUT_CHANNEL,
                               kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=self.params.DQN_CONV1_OUT_CHANNEL,
                               out_channels=self.params.DQN_CONV2_OUT_CHANNEL,
                               kernel_size=kernel_size + 1)
        self.conv3 = nn.Conv2d(in_channels=self.params.DQN_CONV2_OUT_CHANNEL,
                               out_channels=self.params.DQN_CONV2_OUT_CHANNEL,
                               kernel_size=kernel_size + 2)
        # self.fc1 = nn.Linear(in_features=self.params.DQN_CONV2_OUT_CHANNEL*4 + self.params.OBJECT_TYPE_NUM, # +2 for preferences
        #                      out_features=350)
        self.fc1 = nn.Linear(in_features=self.params.DQN_CONV2_OUT_CHANNEL * 4,
                             # +2 for preferences
                             out_features=256)
        # self.fc2 = nn.Linear(in_features=350,
        #                      out_features=256)

        self.fc2 = nn.Linear(in_features=256+ self.params.OBJECT_TYPE_NUM,
                             out_features=192)

        self.fc3 = nn.Linear(in_features=192,
                             out_features=128)

        self.fc4 = nn.Linear(in_features=128,
                             out_features=64)
        # self.fc4 = nn.Linear(in_features=32,
        #                      out_features=64)

        # self.deconv1 = nn.ConvTranspose2d(in_channels=1,
        #                                   out_channels=1,
        #                                   stride=1,
        #                                   kernel_size=3)
        #
        # self.deconv2 = nn.ConvTranspose2d(in_channels=1,
        #                                   out_channels=1,
        #                                   stride=1,
        #                                   kernel_size=2)

    def forward(self, env_map, agent_preference):
        batch_size = env_map.shape[0]

        y = F.relu(self.conv1(env_map))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.flatten(start_dim=1, end_dim=-1)
        # y = torch.concat([y, agent_need], dim=1)
        # y = self.batch_norm(y)
        y = F.relu(self.fc1(y))
        y = torch.concat([y, agent_preference], dim=1)
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = self.fc4(y)

        y = y.reshape(batch_size,
                      self.params.HEIGHT,
                      self.params.WIDTH)
        return y
