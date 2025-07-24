import tensorflow as tf
import numpy as np



def predict_with_model(model, imgpath):

    image = tf.io.read_file(imgpath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [60,60]) # (60,60,3)
    image = tf.expand_dims(image, axis=0) # (1,60,60,3)

    predictions = model.predict(image) # [0.005, 0.00003, 0.99, 0.00 ....]
    predictions = np.argmax(predictions) # 2

    return predictions

if __name__=="__main__":

    img_path = "D:\\VSC\\IntoToTF\\Dataset\\archive\\Test\\2\\01102.png"
    img_path = "D:\\VSC\\IntoToTF\\Dataset\\archive\\Test\\0\\06639.png"

    model = tf.keras.models.load_model('./Models')
    prediction = predict_with_model(model, img_path)

    print(f"Prediction = {prediction}")