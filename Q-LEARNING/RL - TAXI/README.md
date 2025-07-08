# Q-Learning en Taxi Environment
## 1) Descripción del Taxi environment:
### Objetivo:
* El objetivo del agente es recoger al pasajero del lugar donde se encuentra y llevarlo a uno de los paraderos disponibles (posiciones marcadas en rojo, amarillo, verde y azul)

<div align="center">
  <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/Q-LEARNING/RL%20-%20TAXI/IMAGENES/TaxiEnv.JPG">
</div>

### Espacio de acciones:
* El action space es discreto. Así, el agente puede realizar las siguientes acciones:
  - 0: Mover hacia el sur (abajo)
  -  1: Moverse hacia el norte (arriba)
  - 2: Muévete hacia el este (derecha)
  - 3: Muévete hacia el oeste (izquierda)
  - 4: Recogida de pasajeros
  - 5: Dejar al pasajero

### Espacio de observaciones:
* Hay 500 estados discretos ya que hay 25 posiciones de taxi, 5 posibles ubicaciones del pasajero (incluido el caso en el que el pasajero está en el taxi) y 4 ubicaciones de destino.


## 2) Descripción del funcionamiento
Q Learning es un algoritmo **'off-policy'**. Por ese motivo, se utilizaron 2 políticas diferentes para actuar y actualizar la tabla Q:
- **Epsilon policy acting**: Guía el comportamiento del agente --> exploration/exploitation.
  
  A medida que el agente entrena, el valor de epsilon disminuye. Esto ocasiona que al agente pueda 'aplicar' lo aprendido utilizando el Q-value de la Q-Table.

  ![](https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/Q-LEARNING/RL%20-%20TAXI/IMAGENES/EpsilonPolicy.JPG)
  
- **Greedy policy**: Toma la acción con mayor Q-value en el estado siguiente inmediato para actualizar la Q-Table.
  ![](https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/Q-LEARNING/RL%20-%20TAXI/IMAGENES/PolicyUpd.JPG)


## 3) Hiperparámetros
* **Learning Rate**: El learning rate (taza de aprendizaje) controla qué tanto se va a actualizar el valor actual respecto a la 'información nueva' (estado siguiente inmediato).
  - **Valor seleccionado**: 0.05

* **Gamma**: Controla qué tan en cuenta se tomarán las recompensas futuras.
  - **Valor seleccionado**: 0.99

* **Epsilon decay**: Este valor determina cuánto disminuirá el 'epsilon' en cada episodio.
  - **Valor seleccionado**: 0.001 --> Este número fue elegido ya que ocasiona que en la mitad de episodios de entrenamiento, el agente tenga un 50% de probabilidad de decidir si elegir una acción aleatoria (exploration) o elegir una acción en base a los valores de la Q-Table aprendida hasta ese momento (exploitation).

* **Episodios**: El número de episodios es la cantidad de veces que el agente a a 'entrenarse'.
  - **Valor seleccionado**: 10000 --> Este número se eligió debido a que al reiniciar el ambiente (env.reset()), el estado es diferente. Por ese motivo, se necesitan más episodios de entrenamiento para actualizar la Q-Table de manera correcta. Así, cada vez que se origine un nuevo episodio, sea en la posición/estado que sea, el agente pueda elegir el valor más óptimo, ya que la tabla fue entrenada teniendo en cuenta esta inicialización aleatoria de estados cada vez que el ambiente se 'resetea'.
    

## 4) Prueba del agente
A continuación, muestro al agente, en 3 episodios de test, tomando las acciones con mayor valor 'Q' de la Q-Table (state,action pair) que aprendió durante el entrenamiento (click al gif para iniciar la muestra):
<div align="center">
  <img src="https://github.com/DianaMLlamocaZ/REINFORCEMENT_LEARNING/blob/main/Q-LEARNING/RL%20-%20TAXI/IMAGENES/taxi_env.gif">
</div>
