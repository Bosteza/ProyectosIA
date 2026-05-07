# Reporte de Proyectos de Inteligencia Artificial

Este repositorio reúne varios ejercicios prácticos de inteligencia artificial, aprendizaje automático, procesamiento de lenguaje natural y razonamiento probabilístico. Cada carpeta aborda un problema distinto y exige traducir una idea teórica en una implementación funcional, manteniendo el código relativamente pequeño y enfocado.

## Resumen de las tareas

### Degrees

El proyecto `degrees` modela el problema de encontrar la conexión más corta entre dos actores a través de películas compartidas. La tarea principal fue implementar una búsqueda en anchura para encontrar el camino mínimo entre dos personas dentro de un grafo.

Los conocimientos necesarios fueron representación de grafos, búsqueda no informada, manejo de fronteras con colas y reconstrucción de caminos mediante nodos padre.

### Tic-Tac-Toe

El proyecto `tictactoe` implementa la lógica de un juego de tres en raya y un agente capaz de jugar óptimamente usando minimax. Se completaron funciones para determinar turnos, acciones válidas, estados terminales, utilidad y selección de la mejor jugada.

Los conocimientos necesarios fueron juegos adversariales, minimax, recursión, estados de juego, evaluación de utilidad y copias seguras de estructuras de datos.

### Knights

El proyecto `knights` representa acertijos de caballeros y bribones mediante lógica proposicional. La tarea fue construir bases de conocimiento que permitieran a un verificador de modelos deducir quién decía la verdad y quién mentía.

Los conocimientos necesarios fueron lógica proposicional, conectivos lógicos, bicondicionales, implicaciones, restricciones de exclusividad y razonamiento por modelos.

### Crossword

El proyecto `crossword` genera crucigramas como un problema de satisfacción de restricciones. Se implementaron consistencia de nodo, consistencia de arco con AC-3, verificación de asignaciones, heurísticas y backtracking.

Los conocimientos necesarios fueron CSP, dominios, restricciones unarias y binarias, AC-3, backtracking, heurística MRV, heurística de grado y ordenamiento por valor menos restrictivo.

### Traffic

El proyecto `traffic` entrena una red neuronal convolucional para clasificar señales de tránsito usando imágenes del conjunto GTSRB. Se implementó la carga de datos con OpenCV y la definición de un modelo de TensorFlow/Keras.

Los conocimientos necesarios fueron procesamiento de imágenes, arreglos multidimensionales, redes neuronales convolucionales, capas de pooling, capas densas, dropout, clasificación multiclase y entrenamiento supervisado.

### Shopping

El proyecto `shopping` predice si un usuario realizará una compra a partir de datos de navegación. Se implementó la carga y conversión de datos, el entrenamiento de un clasificador k-nearest neighbors y el cálculo de sensibilidad y especificidad.

Los conocimientos necesarios fueron aprendizaje supervisado, preprocesamiento de datos tabulares, clasificación, partición entre entrenamiento y prueba, y métricas de evaluación más informativas que la simple exactitud.

### Heredity

El proyecto `heredity` calcula distribuciones de probabilidad sobre genes y rasgos en familias. La tarea consistió en calcular probabilidades conjuntas, acumularlas y normalizarlas para obtener distribuciones finales por persona.

Los conocimientos necesarios fueron probabilidad condicional, redes bayesianas, enumeración de estados posibles, mutación genética, normalización de distribuciones y actualización acumulativa de probabilidades.

### Parser

El proyecto `parser` analiza oraciones mediante una gramática libre de contexto y extrae fragmentos nominales. Se completó el preprocesamiento de texto, las reglas gramaticales no terminales y la detección de noun phrase chunks.

Los conocimientos necesarios fueron tokenización, gramáticas libres de contexto, árboles sintácticos, parsing con NLTK, frases nominales y recorrido de árboles.

## Conocimientos integradores

Aunque cada proyecto tiene un enfoque distinto, todos comparten una idea central: representar formalmente un problema para que un programa pueda razonar sobre él. En algunos casos la representación es un grafo, en otros una base lógica, un conjunto de restricciones, una distribución probabilística, una matriz de datos o una gramática.

También fue necesario trabajar con conceptos de ingeniería de software: leer especificaciones, respetar interfaces existentes, mantener cambios acotados, probar con datos de ejemplo y validar que las funciones no solo produzcan resultados, sino que lo hagan de acuerdo con las restricciones del problema.

## Dificultades

Una dificultad importante es que muchos de estos proyectos no consisten solo en “programar una fórmula”, sino en entender cómo modelar el problema. Por ejemplo, en `crossword` no basta con probar palabras al azar: hay que entender dominios, restricciones y propagación de información. En `heredity`, el reto está en multiplicar probabilidades correctas bajo distintas condiciones familiares. En `parser`, el desafío no es solo tokenizar texto, sino diseñar reglas que acepten las oraciones necesarias sin volver la gramática demasiado permisiva.

Otra dificultad común es la recursión. Proyectos como `tictactoe`, `crossword` y `degrees` dependen de explorar espacios de posibilidades. Para alguien que empieza, puede ser complejo visualizar cómo una función recursiva avanza, retrocede y conserva el estado correcto.

También aparece una dificultad práctica: las dependencias y el entorno. `traffic`, por ejemplo, requiere TensorFlow, OpenCV y una arquitectura compatible. En máquinas Apple Silicon, usar Python bajo Rosetta o instalar paquetes incorrectos puede producir errores que no tienen relación directa con el código del proyecto. Esto obliga a distinguir entre errores de implementación y problemas del entorno.

## Conclusión

En conjunto, este repositorio ofrece una introducción amplia y práctica a varias áreas fundamentales de la inteligencia artificial. Los proyectos muestran que la IA no es una sola técnica, sino una colección de métodos para buscar, razonar, inferir, clasificar y analizar información.

El valor principal de estos ejercicios está en conectar teoría con implementación. Cada solución requiere comprender la estructura del problema antes de escribir código. Esa transición entre el razonamiento conceptual y una implementación verificable es una de las habilidades más importantes para avanzar en inteligencia artificial.
