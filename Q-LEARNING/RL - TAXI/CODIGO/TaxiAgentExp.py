import gym
import numpy as np
import cv2

#Cargo la q_table
q_table=np.load("taxi_agent_q_table.npy")


#Creo el ambiente de test
env=gym.make("Taxi-v3",render_mode="rgb_array")


state=env.reset()[0]

episodios=3

for episodio in range(episodios):

    end=False

    #Reseteo el environment
    state=env.reset()[0]

    while not end:
        #Tomo la acción con mayor q_value en ese estado
        accion=np.argmax(q_table[state])

        #El agente toma esa acción
        new_state,reward,terminated,truncated,info=env.step(accion)

        end=terminated or truncated

        #Actualizo el estado
        state=new_state

        #Muestro la visualización
        tecla=cv2.waitKey(1000)

        cv2.imshow("Render",env.render()[:,:,::-1])

        if tecla==ord("q"):
            break
