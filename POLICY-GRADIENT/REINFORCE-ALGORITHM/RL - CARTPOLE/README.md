# Reinforce Algorithm
Implementación del algoritmo Reinforce desde cero para el entendimiento de su funcionamiento interno.

# Descripción
Para la implementación de este algoritmo, utilicé el environment CartPole de GYM y un MLP con SoftMax como activation function en la última capa, ya que necesito obtener las probabilidades de cada acción en un estado.

Además, debido a la naturaleza 'Monte Carlo' del algoritmo, *la política se actualiza utilizando trayectorias completas*, a diferencia de algoritmos que emplean el enfoque TD-Approach, donde la política se actualiza cada 'n_steps' y no necesariamente al final de los episodios debido al 'replay buffer'.  

# Arquitectura de la política 
- La política es un MLP, donde el input_size es la dimensionalidad de cada vector que representa un estado:
  - CartPole --> Dimensión de cada estado: [cart_position,cart_velocity,pole_angle,pole_angular_velocity] --> shape=(4,)
- 2 capas lineales seguidas de ReLU activation function para añadir no linealidad.
- 1 capa final lineal con 'n' neuronas, donde 'n' es igual a la cantidad de acciones, seguida por la SoftMax activation function para obtener las probabilidades de cada acción:
  - CartPole --> Acciones: [0,1] --> 2 acciones
  - **NOTA**: Usé la SoftMax activation function para que la suma de probabilidades individuales de las acciones sumen 1 y posteriormente pueda usar un 'sampling' de la distribución para elegir las acciones que tomará el agente e incentivar una política estocástica, clave en *policy methods*.
- Imagen de la arquitectura del modelo:
  
  ![ArquitecturaPolicy](https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/POLICY-GRADIENT/REINFORCE-ALGORITHM/RL%20-%20CARTPOLE/IMAGENES/ArquitecturaModelo.JPG)
# Funcionamiento del algoritmo
Como se mencionó, el algoritmo Reinforce emplea el Approach de Monte Carlo para actualizar la política. Es decir, actualiza la *política al final* de una trayectoria, y no cada 'n_steps' como lo realizarían los algoritmos basados en TD Approach.

- a) **Rewards To Go**: Se almacenan los valores de las recompensas obtenidas en cada paso que da el agente hasta que el episodio termine.
Dichas recompensas representan las 'recompensas futuras esperadas' que espera obtener el agente si toma dicha acción en ese estado siguiendo la política actual, a partir de ese paso de tiempo 't' en adelante. 

![RewardsToGo](https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/POLICY-GRADIENT/REINFORCE-ALGORITHM/RL%20-%20CARTPOLE/IMAGENES/RewardsToGo.JPG)

**NOTA**: La variable *gamma* controla el valor de las recompensas obtenidas, que se usará posteriormente para actualizar la política. Previo a ello, se realizó una normalización de las recompensas para evitar la inestabilidad en el entrenamiento.
