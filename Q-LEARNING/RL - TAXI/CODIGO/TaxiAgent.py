import gym
import numpy as np

#Creo el env
env=gym.make("Taxi-v3",render_mode="rgb_array")



#Para crear la tabla 'Q', veo el espacio de dimensiones: action y observation space
obs_space=env.observation_space.n

action_space=env.action_space.n


#Creo la q table
q_table=np.zeros(shape=(obs_space,action_space))


#Defino la acting policy (si el agente realiza exploration o exploitation)
def epsilon_policy(epsilon,tabla_q,state):
    random_number=np.random.rand()

    #Si random_number es menor que el valor de epsilon --> exploration
    if random_number<epsilon:
        action=env.action_space.sample()

    #Exploitation (la acción con mayor Q-value)
    else:
        action=np.argmax(tabla_q[state])

    return action


#Epsilon decay en cada episodio
def epsilon_decay(epsilon):
    #En cada episodio, el epsilon cae en un 0.001
    epsilon=epsilon-0.001

    return epsilon


#Entrenamiento
l_r=0.05  #Determina qué tanto se 'actualizará' el valor actual con el nuevo valor Q del estado siguiente inmediato
gamma=0.99  #Determina qué tanto se prioriza el valor del action-state futuro inmediato
episodios=10000
epsilon=1 #Inicialmente es 1


for episodio in range(episodios):
    #En cada episodio, el env se debe reiniciar
    state=env.reset()[0]

    #Variable que define el término de un episodio
    end=False

    #Array donde almacenaré las recompensas por episodio
    reward_episodio=[]

    while not end:
        #Se elige una acción
        action=epsilon_policy(epsilon,q_table,state)

        #El agente realiza esa acción en ese estado
        new_state,reward,terminated,truncated,info=env.step(action)

        #Verifico si el agente logró pasar el juego (terminated) o perdió (truncated)
        end=terminated or truncated

        #Actualizo el pair action-state en la Q_Table con la fórmula
        q_table[state,action]=q_table[state,action]+l_r*(reward+gamma*np.max(q_table[new_state])-q_table[state,action])

        #Actualizo el estado al actual
        state=new_state


        reward_episodio.append(reward)

    #print(f"Episodio: {episodio+1}, Reward: {np.mean(np.array(reward_episodio))}")


    #Aquí realizo el epsilon decay en cada episodio
    epsilon=np.max(epsilon_decay(epsilon),0)
    print(f"Epoca: {episodio+1}, Epsilon: {epsilon}")

np.save("taxi_agent_q_table",q_table)

print(f"Q_Table guardada!!")


