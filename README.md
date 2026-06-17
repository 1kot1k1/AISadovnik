🌿 Plant Disease Detection — AI System for Plant Disease Recognition

Python • TensorFlow • Flask • ResNet50 • MobileNetV2 • Computer Vision

Plant Disease Detection — интеллектуальная система распознавания заболеваний растений на основе методов глубокого обучения и компьютерного зрения.

🚀 Ключевые возможности
🤖 Двойная система анализа

Приложение использует две архитектуры нейронных сетей:

ResNet50
MobileNetV2
🔍 Автоматическое распознавание заболеваний

Система определяет заболевание растения по фотографии листа и отображает:

наиболее вероятный класс;
вероятность предсказания;
Top-5 наиболее вероятных диагнозов.
🌱 Поддерживаемые культуры

Модель обучена на датасете PlantVillage и распознаёт:

Картофель (Potato)
Томаты (Tomato)
Перец (Pepper)

Всего: 15 классов заболеваний и здоровых растений.

🛠 Технологический стек
Слой	Технологии
Язык программирования	Python 3
Machine Learning	TensorFlow / Keras
Архитектуры	ResNet50, MobileNetV2
Backend	Flask
Frontend	HTML5, CSS3, JavaScript
Обработка изображений	Pillow (PIL)
Анализ данных	NumPy
Среда обучения	Google Colab
🧠 Архитектура моделей
ResNet50

Глубокая сверточная нейронная сеть с остаточными связями (Residual Connections).

Особенности:

Transfer Learning на базе ImageNet;
Fine-Tuning последних слоев;
Dropout для уменьшения переобучения;
Softmax-классификация на 15 классов.
MobileNetV2

Облегчённая архитектура для быстрого инференса.

Особенности:

Depthwise Separable Convolutions;
Низкое потребление памяти;
Высокая скорость работы;
Transfer Learning.
📊 Результаты обучения
MobileNetV2
Accuracy: 98%
Precision: 98%
Recall: 98%
F1-score: 98%
ResNet50
Accuracy: более 97%
Высокая устойчивость к различным заболеваниям
Стабильные результаты на новых изображениях

Для оценки качества использовались:

Classification Report
Confusion Matrix
Accuracy Curves
Validation Metrics
📂 Структура проекта
models/
├── resnet_weights.weights.h5
├── mobilenet_weights.weights.h5
└── class_names.npy

static/
└── uploads/

templates/
└── index.html

app.py
README.md
💻 Запуск проекта
Установка зависимостей
pip install tensorflow flask pillow numpy
Запуск приложения
python app.py

После запуска откройте:

http://127.0.0.1:5000
📌 Статус проекта

✅ Завершён

Разработана полноценная система распознавания заболеваний растений на основе глубокого обучения с использованием моделей ResNet50 и MobileNetV2. Реализован веб-интерфейс на Flask для загрузки изображений и получения результатов диагностики в режиме реального времени.
