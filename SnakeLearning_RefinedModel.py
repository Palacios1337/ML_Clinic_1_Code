import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class NeuralNetwork(nn.Module):

    #three layers nueral network
    # 16 actions input layer
    # 256 hidden size layer
    # 4 action output layer
    def __init__(self, state_size, action_size):
        super().__init__()
        self.input_layer = nn.Linear(state_size,256)
        self.output_layer = nn.Linear(256,action_size)

    #from what ive seen on pytorch examples this is needed
    #relu was chosen from examples ive seen using NN
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.output_layer(x)
        return x


class ReinforcementLearning:

    def __init__(self, model, learning_rate, discount_rate):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate 
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate) # Optimizer Adam was used
        self.criterion = nn.MSELoss()

    def training(self,initial_state,nextmove,reward,next_state,dead):

        #From examples it needs to be converted into a tensor
        initial_state = torch.tensor(initial_state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        nextmove = torch.tensor(nextmove, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)


        #If the shape is only 1 and these cases are when it is still alive.
        #When dead this is not the case.
        if len(initial_state.shape) == 1:
            initial_state = torch.unsqueeze(initial_state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            nextmove = torch.unsqueeze(nextmove, 0)
            reward = torch.unsqueeze(reward, 0)
            dead = (dead, )

        #reinforcement learning process
        #gets the prediction from the NN
        prediction = self.model(initial_state)

        #gets the close for the use of the next state
        target = prediction.clone()

        #This is dependant on wether its alive or not.
        # V(s) = R(S,a) + discountrate*Max(V(s')
        #Target will then allow changes to the NN.
        for i in range(len(dead)):
            V_new = reward[i]
            if not dead[i]:
                V_new = reward[i] + self.discount_rate * torch.max(self.model(next_state[i]))
                #print(V_new)
            target[i][torch.argmax(nextmove[i]).item()] = V_new


        #This will be applied back into the NN and adjust the weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction) 
        loss.backward()
        self.optimizer.step()













