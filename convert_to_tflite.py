import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'anemia_model_v3.keras')
TFLITE_PATH = os.path.join(BASE_DIR, 'anemia_model.tflite')

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / (1024*1024)
print(f"TFLite model saved: {size_mb:.1f} MB")
print("Done!")
