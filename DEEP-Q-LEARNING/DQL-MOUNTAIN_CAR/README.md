# Deep Q-Learning en Mountain Car
## 1) Descripción del Mountain Car Environment
### **Objetivo**:
* El objetivo es llegar lo más rápido posible a la bandera situada en la cima de la colina derecha, por lo que el agente es penalizado con una recompensa de -1 por cada paso de tiempo.
Es decir, la finalidad del agente es lograr llegar a la cima con la menor cantidad de pasos.
<div align="center">
<img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/DEEP-Q-LEARNING/DQL-MOUNTAIN_CAR/IMAGENES/mountain_car_env.JPG">
</div>

### **Espacio de acciones**:
El action space es discreto. El agente puede tomar las siguientes acciones:
- 0: Acelerar hacia la izquierda
- 1: No acelerar
- 2: Acelerar hacia la derecha

### **Espacio de observaciones**:
El observation space es continuo.
La observación es un ndarray que tiene la forma (2,), donde los elementos corresponden a lo siguiente:
<div align="center">
<img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/DEEP-Q-LEARNING/DQL-MOUNTAIN_CAR/IMAGENES/env_mc.JPG">
</div>

## 2) Descripción del funcionamiento
