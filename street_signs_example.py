from my_utils import split_data, order_test_set, create_generators
from deeplearning_models import streetsigns_model
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

if __name__=="__main__":

    if False:
        path_to_data = "D:\\VSC\\IntoToTF\\Dataset\\archive\\Train"
        path_to_save_train = "D:\\VSC\\IntoToTF\\Dataset\\archive\\training_data\\train"
        path_to_save_val = "D:\\VSC\\IntoToTF\\Dataset\\archive\\training_data\\val"
        split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)

    if False:
        path_to_images = "D:\\VSC\\IntoToTF\\Dataset\\archive\\Test"
        path_to_csv = "D:\\VSC\\IntoToTF\\Dataset\\archive\\Test.csv"
        order_test_set(path_to_images, path_to_csv)

    
    path_to_train = "D:\\VSC\\IntoToTF\\Dataset\\archive\\training_data\\train"
    path_to_val = "D:\\VSC\\IntoToTF\\Dataset\\archive\\training_data\\val"
    path_to_test = "D:\\VSC\\IntoToTF\\Dataset\\archive\\Test"
    batch_size = 64
    epochs = 15
    
    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    TRAIN=False
    TEST=True

    if TRAIN:


        path_to_save_model = './Models'
        ckpt_saver = ModelCheckpoint(
            path_to_save_model, 
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        # After 10 epochs, record val accuracy, if 20 epochs reached then it stops and saves model
        early_stop = EarlyStopping(
            monitor="val_accuray",
            patience=10
        )

        model = streetsigns_model(nbr_classes)

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
            train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[ckpt_saver, early_stop]
        )

    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_generator)

        print("Evaluating test set:")
        model.evaluate(test_generator)