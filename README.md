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


### Documentación

#### Preprocesado
Para la implementación se ha utilizado el DataSet KDDTest. Adicionalmente se han empleado los siguientes parámetros con sus respectivos valores
    a = 0.1
    b = 0.99

Ambos de ellos se emplearán para el rango de la normalización. Posteriormente se emplea el encoder de la librería sklearn donde se proceden a trabajar las tres primeras columnas del dataset. Ahora, se proceden a realizar algunos ajustes dentro del dataset, esto para modificar los valores string y reemplazarlos por valores numéricos dentro del modelo.

    data[41] = data[41].replace(True, 1)    # remplaza valores True por 1
    data.drop(42,axis = 1)                  # se elimina la columna 42
    data.drop(19, axis = 1)                 # se elimina la columna 19 debido a que alberga puros 0

A continuación se deben seleccionar los valores para la matrix X y el vector Y. En este caso se han tomado todas las columanas a excepción de la 41 para X y únicamente la columna 41 para el vector Y. Luego se aplica la normalizaión

    X = data.loc[:, data.columns != 41]           # Matriz X
    y = data.loc[:, data.columns == 41]           # Vector Y
    normalized_X = (X-X.min())/(X.max()-X.min())  # Normalización
    normalized_X = (b-a)*normalized_X+a