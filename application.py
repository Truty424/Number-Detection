import tkinter as tk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
from numpy import argmax
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model
os.environ["PATH"] += ":/usr/local/bin/gs"

window = tk.Tk()

frame = tk.Frame(window)
frame.pack(side=tk.BOTTOM)


label_fin_ans = tk.Label(window, text="Your number is: ")
label_fin_ans.pack(side=tk.BOTTOM, padx=20, pady=20,)

label = tk.Label(window)
label.pack(side=tk.BOTTOM, padx=20, pady=20,  before=label_fin_ans)

canvas = tk.Canvas(bg="white",height=300,width=150)
canvas.pack(anchor='center', padx=30, pady=30, expand=1)

window_width = 512
window_height = 512

button_width = 10
button_height = 2

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

window.geometry('{}x{}+{}+{}'.format(window_width, window_height, x, y))


def cnn_learning():
    label.config(text="Neural Network is currently learning. This may take a moment.")
    label.update()
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    input_shape = (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train = x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test = x_test / 255.0

    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
    batch_size = 64
    num_classes = 10
    epochs = 5

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(epsilon=1e-08), loss='categorical_crossentropy',
                  metrics=['acc'])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') > 0.995:
                print("\nReached 99.5% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        callbacks=[callbacks])
    test_loss, test_acc = model.evaluate(x_test, y_test)

    model.save('final_model.h5')
    label.config(text="The neural network has been successfully trained and is ready for use.")


def predict_number():
    canvas.update()
    canvas.postscript(file="number.ps")

    psimage = Image.open('/Users/adam/PycharmProjects/CNN_firstproject/number.ps')
    psimage = ImageOps.invert(psimage)
    psimage.save('numberPng.png')

    # load the image
    img = load_image('numberPNG.png')
    # load model
    model = load_model('final_model.h5')
    # predict the class
    predict_value = model.predict(img)
    digit = argmax(predict_value)

    last_character = label_fin_ans.cget("text")[-1]
    if last_character.isdigit():
        new_text = label_fin_ans.cget("text")[:-1]
        label_fin_ans.config(text=new_text)

    current_text = label_fin_ans.cget("text")
    new_text = current_text + str(digit)
    label_fin_ans.config(text=new_text)
    print(digit)


def load_image(filename):
    # load the image
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)

    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


def delete_painting():
    canvas.delete("all")
    canvas.update()

    last_character = label_fin_ans.cget("text")[-1]
    if last_character.isdigit():
        new_text = label_fin_ans.cget("text")[:-1]
        label_fin_ans.config(text=new_text)

def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y


def draw_smth(event):
        global lasx, lasy
        canvas.create_line((lasx, lasy, event.x, event.y), fill='black', width=10, capstyle='round')
        lasx, lasy = event.x, event.y


cnn_learning_button = tk.Button(frame, text="Learn CNN", command=cnn_learning, width=button_width, height=button_height)
cnn_learning_button.pack(side=tk.LEFT, pady=20)

draw_button = tk.Button(frame, text="Predict Number", command=predict_number, width=button_width, height=button_height)
draw_button.pack(side=tk.LEFT, pady=20)

delete_drawing_button = tk.Button(frame, text="Delete", command=delete_painting, width=button_width, height=button_height)
delete_drawing_button.pack(side=tk.LEFT, pady=20)

canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw_smth)

window.mainloop()
