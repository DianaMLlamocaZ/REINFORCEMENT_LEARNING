import gym
import cv2
from policy import Modelo
import torch
import numpy as np

env=gym.make("CartPole-v1", render_mode="rgb_array") #Creación del entorno
state_space=env.observation_space.shape[0]
action_space=env.action_space.n


modelo=Modelo(state_space,action_space) #Creación del modelo
weights_model=torch.load("./modelo.pth",weights_only=True)
modelo.load_state_dict(weights_model)


def get_action(state):
    state=torch.tensor(state).unsqueeze(0)

    with torch.no_grad():
        prob_act=modelo(state)

        dist_act=torch.distributions.Categorical(prob_act)

        action=dist_act.sample()

    return action.item()



episode_rewards = []

#Ver cómo actúa en promedio
def evaluation(episodes):
    for episode in range(episodes):
        state=env.reset()[0]
        end=False


        total_reward=0

        while not end:
            with torch.no_grad():

                action=get_action(state)
                
                
                new_state,reward,done,truncated,info=env.step(action)

                end=done or truncated

                state=new_state

                total_reward+=reward

                #Mostrar el render del juego
                frame=env.render()                
                tecla=cv2.waitKey(10)

                cv2.imshow("Render",frame[:,:,::-1])

                if tecla==ord("q"):
                    break
        
        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"\nEvaluación promedio (política estocástica):")
    print(f"Reward promedio: {mean_reward:.2f} ± {std_reward:.2f}")
      

evaluation(10)