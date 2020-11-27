# Tarea 3 Sistemas Distribuidos

Esta es una tarea para la asignatura de Sistemas Distribuidos impartida en la Pontificia Universidad Católica de Valparaíso. El objetivo de esta actividad consta de crear una red neuronal capaz de clasificar entre los diversos tipos de ataques dentro de una red de computadores. Bajo esta premisa se propone el poder crear un sistema de detección de intrusos basado en el dataset KDD.

### Autores

- Gonzalo Tello
- Nicolas Avendaño
- Gonzalo Pauchard
- Rodrigo Maureira

### Configuración

Para asegurar el funcionamiento de este programa, el entorno de Python deberá tenerar las siguientes librerías. Estas se pueden instalar mediante pip de la siguiente manera.

    $ pip install --user pandas
    $ pip install --user numpy
    $ pip install --user sklearn


# Documentación

## Código
### Init()
    Swarm es una matriz de tamano (np, nh*D), cada fila de la matriz representa una matriz de pesos estirada, si por ejemplo son 10 nodos de entrada y 20 escondidos, una matriz de pesos seria de dimension (20,10), entonces una particula representa esta matriz como un solo vector de 20*10 = 200 columnas y una fila, por lo tanto la matriz de la swarm es de num_particulas*200 en este caso de ejemplo.
