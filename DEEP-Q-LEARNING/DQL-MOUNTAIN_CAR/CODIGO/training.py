import gym
import numpy as np
import torch
import cv2
from dqn_network import DQN
from rep_buffer import replay_buffer

#Policy acting greedy
def policy_acting(state,dqn_main,epsilon):
    random_value=torch.rand(size=(1,)).item()

    #Exploration: acción aleatoria
    if random_value<epsilon:
        action=env.action_space.sample()

    #Exploitation: acción con mayor valor 'q'
    else:
        with torch.no_grad():
            res_dqn=dqn_main(torch.tensor(state,dtype=torch.float32)) #Uso la dqn principal (no la target) para seleccionar la acción de mayor q value
            action=torch.argmax(res_dqn).item()

    return action

#Creo el entorno
env=gym.make("MountainCar-v0",render_mode="rgb_array")


#Creo las redes
main_dqn=DQN(states_dim=2,actions_dim=3)
target_dqn=DQN(states_dim=2,actions_dim=3)


#A la target network le coloco los mismos pesos que la main_dqn. Se irá actualizando cada 10 episodios --> Servirá para calcular el TD Target
target_dqn.load_state_dict(main_dqn.state_dict()) 


#Defino los hiperparámetros
learning_rate=1e-3 #1x10''(-3)=0.001
gamma=0.99
epsilon=1
batch_size=64
buffer_size=5000
episodios=1000
target_network_update_ep=10 #1000/10 --> 100 actualizaciones


#Creo el replay buffer
buffer=replay_buffer(length=buffer_size)


#Optimizer main network
opt_main_dqn=torch.optim.Adam(main_dqn.parameters(),learning_rate)


#Loss
loss=torch.nn.MSELoss()


#Aquí almaceno los reward por cada episodio
reward_episodios=[]


#Valor de referncia para guardar el modelo con "mejor recompensa": -200 en mountain car al inicio (hasta que no llegue al flag)
mejor_recompensa_actual=-200 


#Training loop
for episodio in range(episodios):

    state=env.reset()[0]
    end=False

    reward_episodio=0
    

    while not end:
        #Selecciono una acción siguiendo la política policy acting
        action=policy_acting(state,main_dqn,epsilon)
   
        new_state,reward,final,truncated,info=env.step(action) #Se elige esa acción

        end=final or truncated

        #Añado al buffer esa transición
        buffer.push([state,new_state,action,reward,end])

        #Compruebo si el buffer tiene la misma cantidad o más de elementos que el batch_size para iniciar el entrenamiento de la red
        if (buffer.length())>=batch_size:
            #Obtengo una muestra de tamaño == batch_size
            samples_buffer=buffer.sample(batch_dimen=batch_size)

            #Obtengo cada variable por separado usando 'zip'
            states_b,new_states_b,actions_b,rewards_b,end_b=zip(*samples_buffer)
           
            #Convierto a tensores y les añado el batch dimension si es necesario para usarlos en la red
            states_b=torch.tensor(states_b,dtype=torch.float32)  #(64,2)
            new_states_b=torch.tensor(new_states_b,dtype=torch.float32)  #(64,2)
            actions_b=torch.tensor(actions_b,dtype=torch.int64).unsqueeze(1)  #(64,1)
            rewards_b=torch.tensor(rewards_b,dtype=torch.float32).unsqueeze(1)  #(64,1)
            end_b=torch.tensor(end_b,dtype=torch.float32).unsqueeze(1)  #(64,1)
            
            
            #Calculo el TD Target
            with torch.no_grad():
                q_value_max=torch.max(target_dqn(new_states_b),axis=1)[0].unsqueeze(1) #Uso la target network. shape=(64,1)
                td_target=rewards_b+gamma*q_value_max*(1-end_b) #Calculo el td_target para cada transición en el batch
                
            #Obtengo el valor q del 'estado y acción actual' para cada transición del batch (usando la main network)
            res_q_actuales=main_dqn(states_b)
            q_values_actual=res_q_actuales.gather(dim=1,index=actions_b)
            

            #Obtengo el loss error: td_target-q_val_actuales
            loss_error=loss(q_values_actual,td_target)
            

            #Coloco el optimizer a 0 --> Los gradientes serán 0 para evitar acumulación de gradientes en cada iteración
            opt_main_dqn.zero_grad()

            #Calculo los gradientes de los pesos de la red en base al loss error
            loss_error.backward()

            #Actualizo los pesos de la red con step
            opt_main_dqn.step()


        state=new_state #Actualizo el estado actual

        reward_episodio+=reward

    #Epsilon decay
    epsilon=max(0,epsilon*0.999)

    reward_episodios.append(reward_episodio)

    print(f"Episodio {episodio} - Epsilon: {epsilon}")

    #Actualizo los pesos de la target network de acuerdo al valor target network update
    if (episodio%target_network_update_ep==0):
        print("Actualizando la Target Network")
        target_dqn.load_state_dict(main_dqn.state_dict())

    
    #Cada 100 episodios se imprimirá la media de los valores
    if (episodio%100)==0:
        print(f"Episodio {episodio} --- Reward promedio: {np.mean(np.array(reward_episodios))}")
       
    print(f"Episodio: {episodio}. Reward: {reward_episodio}")


    #Guardo los pesos del mejor main network
    if reward_episodio>mejor_recompensa_actual:
        mejor_recompensa_actual=reward_episodio
        torch.save(main_dqn.state_dict(),"weights_main_dqn.pth")

print("Mejor recompensa:",mejor_recompensa_actual)
np.save(file="rewards_episodios",arr=np.array(reward_episodios))
