# GRADIENT METHODS
En este repositorio, implementé algoritmos desde cero de RL que emplean el cálculo de gradientes para actualizar la política.

Implementé los algoritmos 'from scratch' con el objetivo de entender su funcionamiento interno.

# ALGORITMOS
- **Reinforce Algorithm**: Para la implementación de este algoritmo, utilicé el environment CartPole de GYM y un MLP con SoftMax como activation function en la última capa, ya que necesito obtener las probabilidades de cada acción en un estado. Además, debido a la naturaleza 'Monte Carlo' del algoritmo, se utilizan trayectorias para calcular la estimación de 'recompensas' que obtiene el agente al tomar una acción en ese estado.  
