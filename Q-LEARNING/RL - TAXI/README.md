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
