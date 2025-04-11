# Uso del algoritmo Q-Learning en el environment Frozen Lake

## 1) Descripción del environment Frozen Lake
### **Objetivo**:
* El agente debe cruzar un lago congelado desde el inicio hasta la meta sin caer en ningún agujero.
![Environment](https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/Q-LEARNING/RL-FROZEN_LAKE/IMAGENES/Imagen1.JPG)

### **Espacio de acciones**:
* El espacio de acciones es discreto. Son 4 acciones:
  - izquierda (0), abajo (1), derecha (2), arriba (3) 

### **Espacio de observaciones**
* El espacio de observaciones también es discreto. Son 16 observaciones posibles, al ser el mapa 4x4.

### **Recompensas**
* Las recompensas en el environment son las siguientes:
  - alcanzar la meta: +1
  - caer en un agujero: 0
  - no logra la meta: 0
 
## 2) Descripción del funcionamiento
### **Implementación de la política epsilon greedy**
* El valor inicial de 'epsilon' será 1. A medida que se vayan dando más iteraciones, el valor de epsilon decaerá al multiplicársele un factor de decaimiento, lo que provocará que a medida que haya más entrenamiento, el modelo tenderá a 'explotar' lo aprendido. Es decir, preferirá seleccionar la acción que le genera más recompensa según lo aprendido.
