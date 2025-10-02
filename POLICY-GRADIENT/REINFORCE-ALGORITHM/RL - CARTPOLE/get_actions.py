import torch
from policy import Modelo
import gym
import numpy as np
from collections import deque

env=gym.make("CartPole-v1", render_mode="rgb_array") #Creación del entorno

state_space=env.observation_space.shape[0]
action_space=env.action_space.n

modelo=Modelo(state_space,action_space) #Creación del modelo


#obtener acción y el log de la probabilidad de esa acción
def get_action_n_logprobs(state):
   
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_probs = modelo(state)
    
    dist = torch.distributions.Categorical(action_probs)

    action = dist.sample()
    log_action = dist.log_prob(action)
    entropy = dist.entropy()

    return action.item(), log_action, entropy



#Reward to go
def rewards_to_go(rewards,gamma):
    G=0
    returns=[]


    for reward in reversed(rewards):
        G=reward+gamma*G #reward to go recursivamente

        returns.insert(0,G) #index=0 porque el reward to go se calcula 'in reverse'

    return returns



def training():
    #Hiperparámetros
    episodes=1000
    lr=1e-3
    gamma=0.99
    opt=torch.optim.Adam(modelo.parameters(),lr)
    beta=0.1 #0.01

    score_avg=deque([],maxlen=100)


    for episode in range(episodes):
        #El agente aprende de CADA episodio
        rewards=[]
        log_probs=[]
        entropies=[]

        end=False
        state=env.reset()[0]



        #Por cada episodio (monte carlo)
        while not end:
            action,log_action,entropy=get_action_n_logprobs(state) #elijo una acción cuya distribución depende de las probs. de las acciones
            
            new_state,reward,done,truncated,info=env.step(action)

            end=done or truncated

            rewards.append(reward) #para calcular el reward-to-go
            log_probs.append(log_action) #log probs para el gradiente
            entropies.append(entropy) #entropía para evitar el determinismo rápido
            
            state=new_state #Actualizo el estado


        score_avg.append(sum(rewards)) #Por cada episodio terminado o truncated, sumar el reward total

            

        #Calculo el gradiente
        cumulative_reward=torch.tensor(rewards_to_go(rewards,gamma))
        logs_probs_final=torch.stack(log_probs) #stack para NO romper el grafo computacional, NO torch.tensor
        entropies_final=torch.stack(entropies) 
        

        #Para la inestabilidad
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(rewards_to_go(rewards,gamma)).unsqueeze(1)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        

        
        #Se invierte el loss mult. por -1 para el gradient DESCENT en pytorch
        policy_loss=-torch.sum(returns*logs_probs_final) #cumulative_reward
        entropy_bonus=torch.sum(entropies_final) #esto no había
 

        #loss_final=policy_loss-beta*entropy_bonus #beta controla qué tanta aleatoriedad tendrá el agente
        loss_final=policy_loss


        
        #Actualización de los parámetros de la red
        opt.zero_grad() #Grads a cero para los parámetros de la red

        loss_final.backward() #Calculo los gradientes en base al loss

        opt.step() #Actualizo los parámetros de la red
        

        if (episode%100)==0:
            print(f"Episodio {episode} --> Reward average últimos 100 eps: {np.mean(score_avg)}")
            
        
training()


torch.save(modelo.state_dict(),"./modelo.pth")

print("Modelo guardado!")
