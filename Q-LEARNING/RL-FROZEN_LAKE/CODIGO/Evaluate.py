#Evalúo el agente
import gym
import numpy as np
import cv2

#Ambiente
env = gym.make("FrozenLake-v1", render_mode="rgb_array", map_name="4x4", is_slippery=False)

#Cargo la tabla q
q_table=np.load("tabla_q_frozenlake.npy")

#Modo solo exploración
epsilon=0

#Ejecuto un episodio de prueba
obs,info=env.reset()

#Recompensa total
recompensa_total=0

#Variable que indica si ya finalizó
end=False

while not end:
    #Determino la acción que va a tomar (obvio esto es en base a la tabla aprendido)
    accion=np.argmax(q_table[obs])

    #Toma esa acción
    obs,reward,final,truncated,info=env.step(accion)

    #Si finaliza, que el bucle se rompa, y no se muestre más la animación
    end=final or truncated

    #A medida que itera, sumo la recompensa
    recompensa_total+=reward


    #En caso contrario, muestro en pantalla la animación hasta que llegue a la meta o se trunque
    img=env.render()

    cv2.imshow("Render",img)
    if cv2.waitKey(100)==ord("q"):
        break


print(f"La recompensa total en este episodio es {recompensa_total}")
