# Proyectos Policy Gradient - Reinforce Algorithm
Implementación del algoritmo Reinforce desde cero para el entendimiento de su funcionamiento interno.

# Descripción
Para la implementación de este algoritmo, utilicé el environment CartPole de GYM y un MLP con SoftMax como activation function en la última capa, ya que necesito obtener las probabilidades de cada acción en un estado. Además, debido a la naturaleza 'Monte Carlo' del algoritmo, se utilizan trayectorias para calcular la estimación de 'recompensas' que obtiene el agente al tomar una acción en ese estado.  

# Funcionamiento del algoritmo
