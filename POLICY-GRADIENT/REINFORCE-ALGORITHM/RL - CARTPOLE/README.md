# Reinforce Algorithm
Implementación del algoritmo Reinforce desde cero para el entendimiento de su funcionamiento interno.

# Descripción
Para la implementación de este algoritmo, utilicé el environment CartPole de GYM y un MLP con SoftMax como activation function en la última capa, ya que necesito obtener las probabilidades de cada acción en un estado.

Además, debido a la naturaleza 'Monte Carlo' del algoritmo, *la política se actualiza utilizando trayectorias completas*, a diferencia de algoritmos que utilizan el enfoque TD-Approach, donde la política se actualiza cada 'n_steps' y no necesariamente al final de los episodios.  

# Funcionamiento del algoritmo
## Arquitectura de la política 
