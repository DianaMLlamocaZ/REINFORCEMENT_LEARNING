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

# Funcionamiento del algoritmo
