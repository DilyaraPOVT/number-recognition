# number-recognition
Распознавание цифр с графических файлов.
Это задача классификации. Всего 10 классов, столько же, сколько и цифр - от 0 до 9. Наша задача, определить, к какому классу относится изображение.
для решения был использоваан интерпретатор (Google Colab)
____
```
import tensorflow as tf # библотека для глубокого обучения нейронных сетей
import numpy as np # библотека для работы с массивами
import matplotlib.pyplot as plt # библиотека для визуализации данных
import imageio # библиотека для работы с изображениями
```

using MarkdownSharp;
using MarkdownSharp.Extensions.Mal;

Markdown mark = new Markdown();

// Short link for MAL - 
// http://myanimelist.net/people/413/Kitamura_Eri => mal://Kitamura_Eri
mark.AddExtension(new Articles()); 
mark.AddExtension(new Profile());

mark.Transform(text);
