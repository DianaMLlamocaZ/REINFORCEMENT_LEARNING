#Importo las librerías necesarias
import gym
import cv2
import numpy as np


#Obtengo en env
env=gym.make("FrozenLake-v1",render_mode="rgb_array",map_name="4x4",is_slippery=False)


#Determino el valor 'inicial' de epsilon
epsilon=1


#Función que crea la Q table
def q_table():
    q_table=np.zeros(shape=(env.observation_space.n,env.action_space.n))
    return q_table



#Defino la política del epsilon (notar que el espacio de estados ya es discreto)
def epsilon_greedy_policy(obs_ac_disc,epsilon,tabla):
    #Número aleatorio. Rango: [0,1), para que haya probabilidades de que a medida que aumente el valor de las iteraciones, el agente priorice más la explotation que la exploration
    valor_random=np.random.random()

    #Exploration
    if valor_random<epsilon:
        action=env.action_space.sample()

    #Explotation
    else:
        #De lo contrario, se obtiene la acción con el mayor valor en ese estado
        action=np.argmax(tabla[obs_ac_disc])

    return action


def training(learning_rate,discount_rate,epochs):
    #El valor de epsilon que controla la exploración y la explotación del agente en cada iteración
    global epsilon

    #Recompensa
    recompensa_total=[]

    #Creo la tabla q
    tabla_q=q_table()

    #Entreno por la cantidad indicada
    for epoch in range(epochs):

        recompensa_epoca=0

        obs_ac,info=env.reset()

        end=False

        while not end:
            #Uso la política greedy y en base a ello tomo alguna acción
            action=epsilon_greedy_policy(obs_ac,epsilon,tabla_q)

            #Esa acción es tomada por el agente
            obs_new,recompensa,final,truncated,info=env.step(action)

            #Si llegó a la meta o simplemente no logró completarlo luego de x iteraciones (truncated), entonces se rompe el bucle
            end=final or truncated

            
            #De lo contrario, se aplico la fórmula en base a esos valores para actualizar la tabla
            tabla_q[obs_ac,action]=tabla_q[obs_ac,action]+learning_rate*(recompensa+discount_rate*np.max(tabla_q[obs_new])-tabla_q[obs_ac,action])

            #Recompensa por cada iteración hasta que llegue a la meta o se trunque
            recompensa_epoca+=recompensa

            #El estado actual debo actualizarlo
            obs_ac=obs_new

        #Aquí aplico el decaimiento de epsilon. Mientras más alto sea el valor de la iteración, el agente priorizará elegir la acción que tiene mayor q_value en
        #la tabla en ese estado determinado (exploration)
        epsilon=max(0.01,epsilon*0.999)

        #Agrego a la lista de recompensas
        recompensa_total.append(recompensa_epoca)


        #Cada 5 iteraciones, que imprima el promedio de la sumatorioa de recompensas de las últimas 100 épocas
        if (epoch%5)==0:
            print(f"Época {epoch}. Recompensa: {np.mean(recompensa_total[-100:])}")

            letra=cv2.waitKey(1000)

            cv2.imshow("Render",env.render())
            if letra==ord('q'):
                break

    
    return tabla_q

l_r=0.05
d_r=0.9

tabla_final=training(l_r,d_r,6000)
print(tabla_final)

#Guardo la tabla 'q'
np.save("tabla_q_frozenlake.npy",tabla_final)

