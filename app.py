from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os

# ==========================================
# FLASK CONFIG
# ==========================================
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# ЗАГРУЗКА КЛАССОВ
# ==========================================
# Убедись, что файл лежит по этому пути
class_names = np.load("models/class_names.npy", allow_pickle=True)
num_classes = len(class_names)

# ==========================================
# МОДЕЛЬ RESNET50 (Размер 96x96)
# ==========================================
resnet_base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(96, 96, 3)
)
resnet_base.trainable = False

resnet_inputs = layers.Input(shape=(96, 96, 3))
x1 = resnet_base(resnet_inputs, training=False)
x1 = layers.GlobalAveragePooling2D()(x1)
x1 = layers.Dropout(0.4)(x1)
resnet_outputs = layers.Dense(num_classes, activation="softmax")(x1)

resnet_model = models.Model(resnet_inputs, resnet_outputs)
resnet_model.load_weights("models/resnet_weights.weights.h5")
print("✅ ResNet50 загружена (96x96)")

# ==========================================
# МОДЕЛЬ MOBILENETV2 (Размер 128x128)
# ==========================================
mobilenet_base = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(128, 128, 3)
)
mobilenet_base.trainable = False

mobilenet_inputs = layers.Input(shape=(128, 128, 3))
# ВАЖНО: Мы убрали отсюда preprocess_input, чтобы не делать его дважды
x2 = mobilenet_base(mobilenet_inputs, training=False)
x2 = layers.GlobalAveragePooling2D()(x2)
x2 = layers.Dropout(0.3)(x2)
mobilenet_outputs = layers.Dense(num_classes, activation="softmax")(x2)

mobilenet_model = models.Model(mobilenet_inputs, mobilenet_outputs)
mobilenet_model.load_weights("models/mobilenet_weights.weights.h5")
print("✅ MobileNetV2 загружена (128x128)")

# ==========================================
# ГЛАВНАЯ СТРАНИЦА
# ==========================================
@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    results = None
    winner = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        image_path = filepath

        # Открываем оригинал изображения
        raw_img = Image.open(filepath).convert("RGB")

        # --- ПОДГОТОВКА ДЛЯ RESNET (96x96) ---
        img_96 = raw_img.resize((96, 96))
        arr_96 = np.expand_dims(np.array(img_96), axis=0)
        # Стандартная предобработка (Caffe style)
        resnet_input_data = tf.keras.applications.resnet50.preprocess_input(arr_96.copy())
        
        resnet_pred = resnet_model.predict(resnet_input_data, verbose=0)[0]
        resnet_idx = np.argmax(resnet_pred)
        resnet_result = {
            "model": "ResNet50",
            "class": class_names[resnet_idx],
            "confidence": round(float(resnet_pred[resnet_idx]) * 100, 2)
        }

        # --- ПОДГОТОВКА ДЛЯ MOBILENET (128x128) ---
        img_128 = raw_img.resize((128, 128))
        arr_128 = np.expand_dims(np.array(img_128), axis=0)
        # Стандартная предобработка (Scaling to -1...1)
        mobilenet_input_data = tf.keras.applications.mobilenet_v2.preprocess_input(arr_128.copy())
        
        mobilenet_pred = mobilenet_model.predict(mobilenet_input_data, verbose=0)[0]
        mobilenet_idx = np.argmax(mobilenet_pred)
        mobilenet_result = {
            "model": "MobileNetV2",
            "class": class_names[mobilenet_idx],
            "confidence": round(float(mobilenet_pred[mobilenet_idx]) * 100, 2)
        }

        # Сбор и сравнение
        results = [resnet_result, mobilenet_result]
        winner = max(results, key=lambda x: x["confidence"])

    return render_template(
        "index.html",
        image=image_path,
        results=results,
        winner=winner
    )

if __name__ == "__main__":
    # host='0.0.0.0' говорит Flask слушать все сетевые интерфейсы
    app.run(host='0.0.0.0', port=5000, debug=True)а