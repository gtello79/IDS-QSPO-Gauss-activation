# Tarea 3 Sistemas Distribuidos

Esta es una tarea para la asignatura de Sistemas Distribuidos impartida en la Pontificia Universidad Católica de Valparaíso. El objetivo de esta actividad consta de crear una red neuronal capaz de clasificar entre los diversos tipos de ataques dentro de una red de computadores. Bajo esta premisa se propone el poder crear un sistema de detección de intrusos basado en el dataset KDD.

### Autores

- Gonzalo Tello
- Nicolas Avendaño
- Gonzalo Pauchard
- Rodrigo Maureira

### Configuración

Para asegurar el funcionamiento de este programa, el entorno de Python deberá tener las siguientes librerías. Estas se pueden instalar mediante pip de la siguiente manera.

    $ pip install --user pandas
    $ pip install --user numpy
    $ pip install --user sklearn


# Documentación

#### Preprocesado
Para la implementación se ha utilizado el DataSet KDDTest(Train?). Adicionalmente se han empleado los siguientes parámetros con sus respectivos valores
    a = 0.1
    b = 0.99

Ambos de ellos se emplearán para el rango de la normalización. Posteriormente se emplea el encoder de la librería sklearn donde se proceden a trabajar las tres primeras columnas del dataset. 

<<<<<<< HEAD
    data[41] = data[41].replace(True, 1)    # remplaza valores True por 1
    data.drop(42,axis = 1)                  # se elimina la columna 42
<<<<<<< HEAD
    data.drop(19, axis = 1)                 # se elimina la columna 19 debido a que alberga puros 0
=======
>>>>>>> 227c3a2832792f5a508dd263d1fd2b3a3ff7985b
=======
Ahora, se proceden a realizar algunos ajustes dentro del dataset, esto para modificar los valores de tipo string y reemplazarlos por valores numéricos dentro del modelo.
>>>>>>> a7b1456802a6d7e55d3249259a3d78d0fd49bdc1

    data = data.drop(42,axis = 1)                  # #se dropea la ultima columna que representa la dificultad del input
    data[41] = data[41].replace(True, 1)           # remplaza valores True por 1
    

A continuación se deben seleccionar los valores para la matriz X y el vector Y. En este caso se han tomado todas las columanas a excepción de la 41 para X y únicamente la columna 41 para el vector Y. Luego se aplica la normalización

    X = data.loc[:, data.columns != 41]           # Matriz X
    y = data.loc[:, data.columns == 41]           # Vector Y
    normalized_X = (X-X.min())/(X.max()-X.min())  # Normalización
<<<<<<< HEAD
    normalized_X = (b-a)*normalized_X+a
=======
    normalized_X = (b-a)*normalized_X+a

En caso de que se desee preprocesar un archivo diferente al establecido dentro del repositorio, se deberá especificar manualmente modificando la siguiente variable dentro del archivo preproceso.py. A continuación un ejemplo de como debe quedar:

    DATA_PATH = 'Data/KDDTrain+_20Percent.txt'

## Código QPSO

### Init()
<<<<<<< HEAD
    Swarm es una matriz de tamano (np, nh*D), cada fila de la matriz representa una matriz de pesos estirada, si por ejemplo son 10 nodos de entrada y 20 escondidos, una matriz de pesos seria de dimension (20,10), entonces una particula representa esta matriz como un solo vector de 20*10 = 200 columnas y una fila, por lo tanto la matriz de la swarm es de num_particulas*200 en este caso de ejemplo.
>>>>>>> 227c3a2832792f5a508dd263d1fd2b3a3ff7985b
=======
Swarm es una matriz de tamano (np, nh*D), cada fila de la matriz representa una matriz de pesos estirada, si por ejemplo son 10 nodos de entrada y 20 escondidos, una matriz de pesos seria de dimension (20,10), entonces una particula representa esta matriz como un solo vector de 20*10 = 200 columnas y una fila, por lo tanto la matriz de la swarm es de num_particulas*200 en este caso de ejemplo. Luego de definir la matriz de realiza la inicialización de la población con:
    
    self.ini_swarm(numPart,numHidden,D)
    
### Ini_Swarm()

Luego se define, se dimensiona y se rellena la matriz Swarm que corresponde al enjambre del cual se realizará el proceso.
    
### Función de activación (Gaussian):

### Run_QPSO:

Esta es la función que predice el movimiento de las particulas la cual nos permite ajustar el peso de las partículas del enjambre para lograr un valor especificado con anterioridad. Esta logra recomponer las matrices de pesos de cada partícula y permite testear su MSE.
    
### Función Fitness:

Esta función determina que tan "buena" es la posición actual para cada partícula
   


##
>>>>>>> a7b1456802a6d7e55d3249259a3d78d0fd49bdc1
