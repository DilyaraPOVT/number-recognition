Распознавание цифр с графических файлов.    
Это задача классификации. Всего 10 классов, столько же, сколько и цифр - от 0 до 9. Наша задача, определить, к какому классу относится изображение.    
Для решения был использоваан интерпретатор (Google Colab)
____
```
import tensorflow as tf # библотека для глубокого обучения нейронных сетей
import numpy as np # библотека для работы с массивами
import matplotlib.pyplot as plt # библиотека для визуализации данных
import imageio # библиотека для работы с изображениями
```
___    
Загрузка обучающей и тестовой выборок - массив данных MNIST. На обучающей выборке обучаем нейронную сеть, на тестовой - проверяем, насколько хорошо нейронная сеть обучилась.    

```
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
Numpy-массив x_test состоит из 10 000 элементов (это примеры). Каждый элемент представляет из себя другой массив размером 28X28. В массивах x_train, x_test хранятся изображения цифр в виде Numpy-массива, где эти изображения размерами 28X28 пикселей и где каждый пиксель представляет собой яркость этого пикселя от 0 до 255.

Numpy-массив y_test состоит из 10 000 элементов. Здесь хранятся числа - ответы к изображениям (выходы).    
```
print("Изображения цифр (10 000 эл-тов, каждый из которых - массив размером 28X28):")
print(x_test.shape) # изображения цифр в виде Numpy-массива, где эти изображения размерами 28X28 пикселей и где каждый пиксель представляет собой яркость этого пикселя от 0 до 255

print("\nЦифры:")
print(y_test.shape) # числа - ответы к изображениям
```
Отрисовка первых 16-ти цифр. Подписи под изображениями из y_train.
```
print("\nОтрисовка  первых 16-ти цифр. Подписи под изображениями из y_train:")
plt.figure(figsize=(8, 8)) # Создание новой фигуры
for i in range(16):
  plt.subplot(4, 4, i + 1)
  plt.xticks([]) # Получить или установить текущие положения делений и метки оси x
  plt.yticks([]) # Получить или установить текущие положения делений и метки оси y
  plt.grid(False) # Настройка линии сетки - она убирается
  plt.imshow(x_train[i], cmap=plt.cm.binary) # Отображать данные как изображение; т.е. на обычном 2D-растре. Входными данными могут быть фактические данные RGB (A) или 2D скалярные данные, которые будет отображаться как псевдоцветное изображение.
  plt.colorbar() # Добавление шкалы цвета к графику
  plt.xlabel(y_train[i]) # Добавление метки для оси x - ответы
plt.show()
```
![image](https://user-images.githubusercontent.com/56718341/129435609-2bc023bc-56e8-48d9-b598-19cffdacbb1d.png)

Для обучения нейронной сети приведем данные к нормальному виду: яркость пикселей должна измеряться от 0 до 1, поэтому каждый из массивов разделим на 255.
```
x_train = x_train / 255
x_test = x_test / 255
```
Создаем модель нейроной сети из двух слоев, где первый будет слоем трансормации в новую размерность. На вход подаем один пареметр - размер нашего массива. Этот слой распрямит массив и превратит в одномерный массив длиной 28X28.

Второй слой принимает на себя два параметра - первый это кол-во нейронов (10 нетйронов, где каждый нейрон будет соответсовать классу цифр от 0 до 10), второй параметр - функция активации SoftMax, скрытый слой (512 нейронов)

```
model = tf.keras.models.Sequential([
       tf.keras.layers.Conv2D(
           input_shape=(28, 28, 1),
           filters=32,
           kernel_size=(5, 5),
           padding='same',
           activation='relu'
       ),
       tf.keras.layers.MaxPool2D(pool_size=(2, 2)), 
       tf.keras.layers.Conv2D(
           input_shape=(28, 28, 1),
           filters=64,
           kernel_size=(5, 5),
           padding='same',
           activation='relu'
       ),
       tf.keras.layers.MaxPool2D(pool_size=(2, 2)),                             
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(1024, activation=tf.nn.relu),
       tf.keras.layers.Dense(10, activation=tf.nn.softmax)                             
])
```
После создания архитектуры модели нейронной сети необходимо модель скомпилировать. Она будет принимать 3 параметра - оптимизатор, функция потерь (будет считать ошибку нейронной сети при обучении с учителем), метирика (будем считать точность)
```
model.compile(
    optimizer='adamax',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
Обучение модели нейронной сети. Также есть 3 параметра - тренировочный набор данных x_train, ответы на изображения массив y_train, кол-во эпох (то кол-во раз, сколько нейронная будет прогонять через себя тренировочный набор данных).
```
model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=13)
```
![image](https://user-images.githubusercontent.com/56718341/129435661-da5020e5-75c1-44e0-816d-209efafc1004.png)

Проверка качества обучения нейронной сети на тестовом наборе. Всего 2 параметра - тестовый массив с изображениями и тестовый массив с цифрами.

```
print("\nПроверка качества обучения нейронной сети на тестовом наборе:")
print(model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test))
```
![image](https://user-images.githubusercontent.com/56718341/129435690-2ee67459-032e-48ae-b035-b7690979019e.png)

Загрузка в нейронную сеть изображение цифры. Для этого напишем отдельную функцию model_answer, которая принимает 3 параметра - обученная модель нейронной сети, имя файла (имя изображения, который загружаем), отрсиовка наших изображений.
```
def model_answer(model, filename, display=True):
  image = imageio.imread(filename)
  image = np.mean(image, 2, dtype=float) # пребразование изображения
  image = image / 255
  if display:
    # отрисовка изображения
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(filename)
    plt.show()

  image = np.expand_dims(image, 0)
  image = np.expand_dims(image, -1)
  print (model.predict(image))
  #print (model.predict.np.argmax(image))
  return np.argmax(model.predict(image))
```

```
for i in range(10):
  filename = f'{i}.png'
  print(filename)
  print('Имя файла: ', filename, '\tОтвет сети: ', model_answer(model, filename, False))
  
print(model_answer(model, '4.png')) # отрисовка
```
![image](https://user-images.githubusercontent.com/56718341/129435707-18d5ed61-8e5c-44f3-bb78-bd69ae89d32f.png)
![image](https://user-images.githubusercontent.com/56718341/129435712-a46a3ac1-a47d-4a92-b588-2eae00e20102.png)
