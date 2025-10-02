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
  <div align="center">
  <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/POLICY-GRADIENT/REINFORCE-ALGORITHM/RL%20-%20CARTPOLE/IMAGENES/ArquitecturaModelo.JPG">
  </div>
# Funcionamiento del algoritmo
Como se mencionó, el algoritmo Reinforce emplea el Approach de Monte Carlo para actualizar la política. Es decir, actualiza la *política al final* de una trayectoria, y no cada 'n_steps' como lo realizarían los algoritmos basados en TD Approach.

- ### **a) Rewards To Go**:
  - Se almacenan los valores de las recompensas obtenidas en cada paso que da el agente hasta que el episodio termine.
Dichas recompensas representan las 'recompensas futuras esperadas' que espera obtener el agente si toma dicha acción en ese estado siguiendo la política actual, a partir de ese paso de tiempo 't' en adelante. 
<div align="center">
<img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/POLICY-GRADIENT/REINFORCE-ALGORITHM/RL%20-%20CARTPOLE/IMAGENES/RewardsToGo.JPG">
</div>

- - **NOTA**: La variable *gamma* controla el valor de las recompensas obtenidas, que se usará posteriormente para actualizar la política. Previo a ello, se realizó una normalización de las recompensas para evitar la inestabilidad en el entrenamiento.

- ### **b) Selección de acciones**:
  - A cada acción se le atribuye una probabilidad, debido a la SoftMax activation function en la última capa.
  - Ya que la suma de probabilidades individuales es 1, entonces se tiene una distribución de probabilidad.
De dicha distribución de probabilidad, se samplearán las acciones, pero no será un 'sampleo' completamente aleatorio, sino de acuerdo a las probabilidades individuales.
  - De esta forma, se introduce estocasticidad y también se fomenta el balance entre *exploration/explotation* en un paso, ya que es un algoritmo Off-Policy.
<div align="center">
<img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/POLICY-GRADIENT/REINFORCE-ALGORITHM/RL%20-%20CARTPOLE/IMAGENES/Actions_LogActions.JPG">
</div>

-  - **NOTA**: Esta función devuelve la acción y el log de la probabilidad de la acción (importantes para actualizar la política)
 
- ### **c) Entrenamiento --> Fórmulas utilizadas**:
  - **c.1) Cálculo del gradiente**: Para calcular el gradiente en *una trayectoria*, usé la siguiente fórmula:
  
  <div align="center">
  <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/POLICY-GRADIENT/REINFORCE-ALGORITHM/RL%20-%20CARTPOLE/IMAGENES/GradientUpdate.JPG">
  </div>
  
  - - **NOTA**: La actualización de la política la realicé al final de cada trayectoria.
   
  - **c.2) Manejo de la inestabilidad en el entrenamiento**: Para manejar la inestabilidad durante el entrenamiento, quizás por valores numéricos grandes de las recompensas en comparación a los pesos del modelo, normalicé las recompensas antes de calcular el gradiente:
    <div align="center">
    <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/POLICY-GRADIENT/REINFORCE-ALGORITHM/RL%20-%20CARTPOLE/IMAGENES/Inestiblidad_Manejo.JPG">
    </div>

  - **c.3) Entrenamiento final**: Ya que las recompensas fueron normalizadas, y se tienen las log probabilidades para cada una de las acciones tomadas por el agente en esa trayectoria, calculo el gradiente multiplicando por -1 al loss, para convertir el problema de 'maximización de recompensas', a un problema de 'minimización de pérdida' (son equivalentes, y más viable en frameworks de DeepLearning):
    
  <div align="center">
  <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/POLICY-GRADIENT/REINFORCE-ALGORITHM/RL%20-%20CARTPOLE/IMAGENES/GradientDescent.JPG">
  </div>
  <div align="center">
  <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/POLICY-GRADIENT/REINFORCE-ALGORITHM/RL%20-%20CARTPOLE/IMAGENES/LossFinal.JPG">
  </div>

  - **c.4) Hiperparámetros utilizados**: A continuación, muestro los hiperparámetros que usé para entrenar al modelo
    - ```
      #Hiperparámetros
      episodes=1000
      lr=1e-3
      gamma=0.99
      opt=torch.optim.Adam(modelo.parameters(),lr)
      ```

# Prueba de la política entrenada:
A continuación, muestro la política ya entrenada, funcionando en 10 episodios de prueba en el entorno CartPole.
<div align="center">
<img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/POLICY-GRADIENT/REINFORCE-ALGORITHM/RL%20-%20CARTPOLE/IMAGENES/cartpole_result.gif">
</div>
