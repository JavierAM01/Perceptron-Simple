# Perceptron-Simple


## Índice


## Enunciado


## Introducción 

El perceptrón es un modelo de aprendizaje automático que consiste en una función matemática que toma varias entradas numéricas y produce una salida binaria. Es decir, dado un vector de entrada $x$, el perceptrón calcula la suma ponderada de sus componentes, la cual se compara con un umbral (o threshold) para producir la salida. Si la suma ponderada es mayor que el umbral, la salida es 1; de lo contrario, la salida es 0. 

El perceptrón puede ser visto como un clasificador lineal que divide el espacio de entrada en dos regiones separadas por un hiperplano. La idea básica es ajustar los pesos de la función para que la salida sea lo más cercana posible a la salida deseada para cada vector de entrada. Esto se logra mediante el algoritmo de aprendizaje del perceptrón, que actualiza los pesos de acuerdo con la regla de Hebb, es decir, incrementando o disminuyendo los pesos en función de si la salida es correcta o no. 

El perceptrón es un modelo simple pero poderoso que ha sido utilizado en una gran variedad de aplicaciones, desde la clasificación de dígitos escritos a mano hasta la detección de spam en correos electrónicos. Aunque es limitado en cuanto a su capacidad para resolver problemas más complejos, el perceptrón sentó las bases para el desarrollo de modelos más sofisticados de redes neuronales artificiales.

## Material usado

 Como lenguaje de programacióno usamos python. En cuanto a materiales externos no es necesario ninguno. Únicamente necesitaremos los datos del enunciado y para estudios extra generaremos datos aleatorios con la librería *numpy.random*. 
 
 ## Resultados y conclusiones
 
 Comentar el hecho de que 3 puntos son muy pocos para poder hacer un entrenamiento en un perceptrón y más si únicamente hacemos una *pasada* por cada uno de ellos. El caso es que por cada input o entrada el perceptrón realiza un cálculo lineal en función esta, posteriormente es pasada (en nuestro caso) por la función sigmoide, $sigmoid : \mathbb{R} \rightarrow (0,1)$ definida por

$$ 
sigmoid(x) = \dfrac{1}{1 + e^{-x}} 
$$

la cual genera resultados en el intervalo (0,1). Así podemos calcular el error frente al resultado esperado 0 o 1. 

### Pregunta 1

Para la creación del perceptrón hay que tener en cuenta la inicialización de los pesos, así como la asignación del factor de aprendizaje, $\alpha$. Para ello en el __init__ pasamos como argumentos dicho factor y además el número de inputs que va a tener dicho perceptrón. En este caso va a ser siempre 2 pues vamos a trabajar en el plano, pero este cambiará dependiendo del problema en cuestión.

```python
class Perceptron:
    def __init__(self, n_inputs, lr):
        self.n = n_inputs
        self.lr = lr
        self.w = np.random.randn(n_inputs) # nº aleatorios distribuidos por una : N(0,1)
        self.w0 = np.random.randn(1)
```

Para evaluar un punto en el perceptron, se ejecuta la función lineal característica del perceptrón:

```python
    def forward(self, X):
        pred = np.dot(X, self.w) + self.w0
        return sigmoid(pred)
```

Además hemos de añadirle una función para el entrenamiento del mismo. Para ello utilzaremos la Regla Delta generalizada:

```python
    def fit(self, x, y):
        pred = self.forward(x)
        error = y - pred
        self.w  += self.lr * error * x
        self.w0 += self.lr * error
        return abs(error)
```

 
