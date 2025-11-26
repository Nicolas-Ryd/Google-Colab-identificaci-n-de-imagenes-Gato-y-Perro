# Google-Colab-identificaci-n-de-imagenes-Gato-y-Perro
En este proyecto he creado una IA capaz de identificar y clasificar imagenes de perros y de gatos utilizando machine learning como principal programa y google colab como el compilador para toda la información de cientos de imagenes. 

En primer lugar, se habilitamos el modo legacy Keras, después, el programa carga el modelo ya entrenado, junto a las etiquetas clasificatorias de cada clase.
Una vez que la imagen está transformada, se organiza en un lote de datos, ya que el modelo espera recibir las imágenes en colecciones, aunque solo sea una. Se ejecuta entonces el proceso de predicción, en el que el modelo analiza la imagen y genera una probabilidad para cada clase. Finalmente, se identifica cuál es la clase con mayor probabilidad y se muestra su nombre y el nivel de confianza.

Para cada archivo, deduce la etiqueta esperada basándose en su nombre, prepara la imagen y solicita una predicción al modelo. Si la predicción coincide con la etiqueta real, se considera un acierto; si no, se registra como un error. Al mismo tiempo, se acumulan las probabilidades de acierto para calcular una media final.
