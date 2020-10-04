import tensorflow as tf
import datetime
import yaml
import json
import time
#import neptune
#from neptunecontrib.monitoring.keras import NeptuneMonitor
import wandb
from wandb.keras import WandbCallback
from mymodel import create_model
import os
from tensorflow import keras
import shutil

wandb.init(project="mnist")

params = yaml.safe_load(open('params.yaml'))
epochs = params['epochs']
log_file = params['log_file']
dropout = params['dropout']
dvc_logs_dir = params['dvc_logs_dir']
logs_subdir = 'logs'

#neptune.init('dmpetrov/sandbox')
#neptune.create_experiment(name='exp1', params=params)

mnist = tf.keras.datasets.mnist

class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        shutil.rmtree(dvc_logs_dir, ignore_errors=True)
        os.mkdir(dvc_logs_dir)
        os.mkdir(os.path.join(dvc_logs_dir, logs_subdir))

    def on_epoch_end(self, epoch, logs=None):
        logdir = os.path.join(dvc_logs_dir, logs_subdir)
        for k, v in logs.items():
            file = os.path.join(logdir, k + '.tsv')
            fd = None
            try:
                if not os.path.exists(file):
                    fd = open(file, 'w+')
                    fd.write(str(k) + os.linesep)
                else:
                    fd = open(file, 'a')
                fd.write(str(v) + os.linesep)
            finally:
                if fd:
                    fd.close()

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = create_model(dropout)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

csv_logger = tf.keras.callbacks.CSVLogger(log_file)

start_real = time.time()
start_process = time.process_time()
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[
                        csv_logger,
                        tensorboard_callback
                        #, NeptuneMonitor()
                        , WandbCallback()
                        , CustomCallback()
                    ])
end_real = time.time()
end_process = time.process_time()

with open("summary.json", "w") as fd:
    json.dump({
        "accuracy": float(history.history["accuracy"][-1]),
        "loss": float(history.history["loss"][-1]),
        "time_real" : end_real - start_real,
        "time_process": end_process - start_process
    }, fd)

model.save(os.path.join(wandb.run.dir, "model.h5"))

