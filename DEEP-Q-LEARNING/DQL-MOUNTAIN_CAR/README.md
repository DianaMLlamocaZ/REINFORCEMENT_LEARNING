# Deep Q-Learning en Mountain Car
## 1) Descripción del Mountain Car Environment
### **Objetivo**:
* El objetivo es llegar lo más rápido posible a la bandera situada en la cima de la colina derecha, por lo que el agente es penalizado con una recompensa de -1 por cada paso de tiempo.
Es decir, *la finalidad del agente es lograr llegar a la cima con la menor cantidad de pasos.*
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

-----

## 2) Descripción del funcionamiento
Deep Q-Learning es un algoritmo **off-policy**, ya que usa 2 políticas distintas para decidir qué acción tomar (exploration/exploitation) y cómo calcular el TD-Target.
- **Epsilon policy acting**:
    - Esta política guía al agente para dirigir su comportamiento sobre *cómo elegir la acción*. Si decidir una acción aleatoria o si selecciónar la acción que tiene un mayor q value según el output de la main DQN network.
    - A medida que el agente entrena, el valor de epsilon disminuye, incentivándolo a que pueda 'reforzar' los conocimientos adquiridos durante la etapa de exploración.

  ![](https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/Q-LEARNING/RL%20-%20TAXI/IMAGENES/EpsilonPolicy.JPG)

<br>

- **Greedy policy**:Se usa una política "greedy" para seleccionar la 'siguiente' mejor acción del estado inmediato según la target network.
<div align="center">
  <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/DEEP-Q-LEARNING/DQL-MOUNTAIN_CAR/IMAGENES/greedy_img.JPG">
</div>

- **Algoritmo Deep Q-Learning**:
    - a) **Se implementan 2 redes para estabilizar el entrenamiento: main y target networks**:
        - Cada "k" pasos (hiperparámetro), la target network se actualizará. Es decir, tendrá los mismos pesos que la main network con el objetivo de que el TD Target no 'cambie' constantemente (lo que sucedería si solo se tiene 1 red) y poder estabilizar el training.

<br>

<div align="center">
<img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/DEEP-Q-LEARNING/DQL-MOUNTAIN_CAR/IMAGENES/networks.JPG">
</div>

  - b) **Se crea un replay buffer**:
      - El replay buffer, de tamaño 'size' (hiperparámetro) servirá para almacenar las transiciones obtenidas en cada paso que da el agente. El objetivo de esta 'memoria' es hacer más 'eficiente' el entrenamiento almacenando y eligiendo muestras (cantidad de muestras igual al batch_size) de manera aleatoria para reutilizar algunas durante el entrenamiento y romper la correlación entre experiencias.

<br>
 
- c) **TD-Target**:
    - Se calcula el valor objetivo usando la target network:
      <br>
      <div align="center">
          <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/DEEP-Q-LEARNING/DQL-MOUNTAIN_CAR/IMAGENES/td_target.JPG">
      </div> 

<br>

- d) **TD-Error**:
    - Se calcula restando el TD-Target y el 'q' value para el estado y acción actuales usando la main network:
      <br>
      <div align="center">
      <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/DEEP-Q-LEARNING/DQL-MOUNTAIN_CAR/IMAGENES/td_error.JPG">
      </div>

----

### 3) Pseudocódigo del algoritmo:
A partir del siguiente pseudocódigo, realicé el entrenamiento del agente. 
<div align="center">
<img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/DEEP-Q-LEARNING/DQL-MOUNTAIN_CAR/IMAGENES/pseudocode.JPG">
</div>

----

### 4) Hiperparámetros:
Luego de probar diferentes valores para los hiperparámetros, opté por los siguientes ya que el agente obtuvo un mejor performance.
``` learning_rate=1e-3 #1x10''(-3)=0.001
    gamma=0.99
    epsilon=1
    batch_size=64
    buffer_size=5000
    episodios=1000
    target_network_update_ep=10 #1000/10 --> 100 actualizaciones
```
---

### 5) Resultados en los episodios de test:
- **Número de episodios de prueba**: 10
- **Gráfica del número de 'pasos' en cada episodio**:
    - Mientras 'menos negativo' sea, significa que el agente necesitó menos pasos para llegar a la meta (su objetivo es llegar a la meta con el menor número de pasos posibles).
      La 'única' recompensa que tiene el agente es cuando llega a la cima de la colina (reward=0); de lo contrario, -1. Por ese motivo, el objetivo del agente es que logre un valor menos negativo de los 200 pasos permitidos por episodio.
  <div align="center">
      <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/DEEP-Q-LEARNING/DQL-MOUNTAIN_CAR/IMAGENES/ep_pruebas.JPG">
  </div>
  
- **Promedio total de los 10 episodios**:
    - Media de pasos: 115.6 --> Significa que, en promedio, el agente necesita 115 pasos para llegar a la meta.

---

### 6) Prueba visual:
A continuación, muestro un gif del agente que entrené tratando de alcanzar la meta en 8 episodios de test (click sobre el gif para iniciar la muestra):
<div align="center">
    <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/DEEP-Q-LEARNING/DQL-MOUNTAIN_CAR/IMAGENES/mountain_car.gif">
</div>
