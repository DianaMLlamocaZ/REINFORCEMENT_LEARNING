import torch
import cv2
import gym
from dqn_network import DQN
import matplotlib.pyplot as plt
import numpy as np

weights_model=torch.load("weights_main_dqn.pth")

modelo=DQN(2,3)
modelo.load_state_dict(weights_model)

env=gym.make("MountainCar-v0",render_mode="rgb_array")

reward_episodes=[]

episodios=10

for episodio in range(episodios):
    end=False
    state=env.reset()[0]
    reward_episodio=0
    while not end:
        state=torch.tensor(state,dtype=torch.float32).unsqueeze(0) #Añado el batch dimension
        
        action=torch.argmax(modelo(state)).item()
        
        new_state,reward,final,truncated,info=env.step(action)

        end=final or truncated

        state=new_state

        reward_episodio+=reward

        tecla=cv2.waitKey(10)

        frame=env.render()[:,:,::-1]

        cv2.imshow("Render",frame)
        
        
        if tecla==ord("q"):
            break

    print(f"Episodio {episodio}")
    reward_episodes.append(reward_episodio)
    print(f"Pasos para llegar a la meta: {-1*reward_episodio}")

print(f"Media de pasos: {-1*np.mean(np.array(reward_episodes))}")
plt.plot(list(range(episodios)),np.array(reward_episodes))
plt.show()
