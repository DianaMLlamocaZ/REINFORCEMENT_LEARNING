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

----

## 2) Descripción del funcionamiento
### **Implementación de la política epsilon greedy**
* El valor inicial de 'epsilon' será 1. A medida que se vayan dando más iteraciones, el valor de epsilon decaerá al multiplicársele un factor de decaimiento, lo que provocará que a medida que haya más entrenamiento, el modelo tenderá a 'explotar' lo aprendido. Es decir, preferirá seleccionar la acción que le genera más recompensa según lo aprendido.

### **Cantidad de épocas o iteraciones en el entrenamiento**
* La cantidad de iteraciones o épocas que usé para el entrenamiento fue de 6000. Este valor es variable y dependerá de qué tan bien el agente converge al objetivo final.
  Cada iteración, termina cuando el agente llega a la meta, cae en un agujero o simplemente agota sus posibilidades de movimiento en cada iteración.
Además, es importante mencionar que en cada iteración, el environment se reiniciará para que el agente pueda aprender de experiencias 'independientes'. En otras palabras, cada iteración es una oportunidad que tiene el agente para que explore el entorno desde cero y aprenda nuevas experiencias.

### **Hiperparámetros**
* Learning rate: El learning rate determina qué tanto se actualiza el valor Q (valor de la acción-estado) en cada paso del aprendizaje.
  
  - *Learning rate escogido*: 0.05
  
* Discount rate: Este hiperparámetro determina qué tanto prioriza el agente la acción futura inmediata.
  
  - *Discount rate escogido*: 0.9
  
* Epsilon decay: Este valor determina qué tanto decae el valor de epsilon. Mientras más pequeño sea el valor de epsilon, menos probabilidad de exploración tendrá el agente, priorizando así que el agente permita 'explotar' lo aprendido.
  
  - *Epsilon decay escogido*: 0.999

----

## 3) Evaluación

