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

# Instrucciones

En primera instancia el usuario debiese generar un preproceso de dataset. Esto es limpiar el documento de entrada a fin de limpiar la data y normalizar cada uno de los parámetros. Adicionalmente, se debe especificar su ruta en el documento preproceso.py. Concretamente las variables DATA_PATH y OUT_PATH

    DATA_PATH = /ruta/al/documento.csv
    OUT_PATH = /ruta/documento/salida.txt
    
Para entrenar la red se deben editar los parámetros del archivo param_config.csv. Tal que se respeten las siguientes indicaciones del formato. A continuación la descripción de cada columna:

    L : nodos de la capa escondida
    C : penalidad pseudo inversa
    maxIter : total de iteraciones
    numPart : total de particulas
    N : numero de datos del dataset

Una vez realizado este procedimiento, se deben ejecutar el script train.py, para realizar el entrenamiento de la red. Luego, se puede ejecutar el archiv test.py para llevar a cabo la prueba. Notese que usted deberá detallar el archivo a utilizar dentro de la variable DATA_PATH de script (al igual que en el caso anterior). Para ejecutar basta con emplear:

    $ python train.py
    $ python test.py


# Documentación

#### Preprocesado
Para la implementación se ha utilizado el DataSet KDDTrain. Adicionalmente se han empleado los siguientes parámetros con sus respectivos valores
    a = 0.1
    b = 0.99

Ambos de ellos se emplearán para el rango de la normalización. Posteriormente se emplea el encoder de la librería sklearn donde se proceden a trabajar las tres primeras columnas del dataset. Ahora, se proceden a realizar algunos ajustes dentro del dataset, esto para modificar los valores de tipo string y reemplazarlos por valores numéricos dentro del modelo.

    data[41] = data[41].replace(True, 1)    # remplaza valores True por 1
    data.drop(42, axis = 1)                  # se elimina la columna 42


A continuación se deben seleccionar los valores para la matrix X y el vector Y. En este caso se han tomado todas las columanas a excepción de la 41 para X y únicamente la columna 41 para el vector Y. Luego se aplica la normalizaión

    X = data.loc[:, data.columns != 41]           # Matriz X
    y = data.loc[:, data.columns == 41]           # Vector Y
    normalized_X = (X-X.min())/(X.max()-X.min())  # Normalización
    normalized_X = (b-a)*normalized_X+a


En caso de que se desee preprocesar un archivo diferente al establecido dentro del repositorio, se deberá especificar manualmente modificando la siguiente variable dentro del archivo preproceso.py. A continuación un ejemplo de como debe quedar:

    DATA_PATH = 'Data/KDDTrain+_20Percent.txt'

## Código QPSO 

### Init()

Esta funcion permite inicializar el enjambre de particulas y sus principales parámetros son:

    MaxIter: Corresponde al total de iteraciones
    NumPart: Corresponde al número de partículas del enjambre
    NumHidden: Número de capas ocultas
    D: Dimensión
    xe
    ye
    C
    
Luego estos son inicializados y se procede a inicializar el enjambre: 
    
     self.ini_swarm(numPart,numHidden,D)
     
### Ini_Swarm()

Esta funcion permite inicializar el enjambre de partículas y sus principales parámetros son: 
        
Swarm es una matriz de tamaño nh x D, (np, nh*D), cada fila de la matriz representa una matriz de pesos estirada, si por ejemplo son 10 nodos de entrada y 20 escondidos, una matriz de pesos seria de dimension (20,10), entonces una particula representa esta matriz como un solo vector de 20*10 = 200 columnas y una fila, por lo tanto la matriz de la swarm es de num_particulas*200 en este caso de ejemplo.

Luego se crea un peso aleatorio para cada partícula y luego se guarda la matriz generada.

### Gaussian_Activation() (Función de Activación)

A cada valor de la matriz, en caso de ser X se transforma en e^(-x^2), por ejemplo si es 2 pasa a ser e^(-4), decidimos usar esta ya que la que poseíamos (e^(-0.5z^2)) tardaba demasiado en realizarse y nosotros usamos el valor directo en vez de usar z, lo cual es la misma implementación realizada de otro modo.

### Run_QPSO

Se carga la función del modelo QPSO para ajustar los pesos de las partículas de nuestro enjambre y así acercarnos a un valor especificado, tras esto se recomponen las matrices de pesos de cada partícula y luego se testea su MSE.

### Fitness() (Función Fitness)

Esta función va a determinar que tan buena es la posición actual para cada partícula evaluada.

## Código Test

Se carga la data para el testeo. Luego se cargan los pesos entrenados y la configuración de los parámetros:

    container = np.load("pesos.npz")
    PARAM_CONFIG_PATH = "param_config.csv"
    params = pd.read_csv(PARAM_CONFIG_PATH)
    
### metrica()

Esta funcion permite visualizar que tan preciso es nuestro algoritmo en base al dato entregado y al dato esperado, esta permite comparar los falsos positivos, los verdaderos negativos, etc, y con todo esto se saca la métrica.

##  Código Train

Aquí se cargan los parametros desde el PARAM_CONFIG_PATH y la data del DATA_PATH, después selecciona los parametros, N que corresponde al número de datos y selecciona de la data los primeros n datos (10.000 por defecto) de X e Y. Luego almacena n y D, que son los tamaños de los datos, para transformarlos (de pandas a numpy) y se suma una columna de "unos" a la X para la bias. 

Por último, para la X se inicializa el QPSO junto con el enjambre y despues se corre la funcion de QPSO para encontrar los pesos, luego de encontrar los pesos ideales se guardan en un archivo llamado "Pesos", los cuales nos permitirán testear en otro archivo.

