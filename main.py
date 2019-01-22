import pandas as pd

from functional_api import MNISTmodel

from subprocess import check_output
print(check_output(['ls', './input']).decode('utf8'))

train = pd.read_csv('./input/train.csv')
test  = pd.read_csv('./input/test.csv')

# hyperparams
input_shape = (28, 28, 1)
batch_size = 64
epochs = 4
number_of_classes = 10

mnist = MNISTmodel(train, test, 
                    input_shape=input_shape, 
                    batch_size=batch_size,
                    epochs=epochs,
                    number_of_classes=number_of_classes)
mnist.build()
mnist.compile()
mnist.summary()

datagen = MNISTmodel.datagen()
datagen.fit(mnist.x_train)

history = mnist.model.fit_generator(datagen.flow(mnist.x_train, mnist.y_train, batch_size=mnist.batch_size),
                                  epochs=mnist.epochs,
                                  verbose=1,
                                  validation_data=(mnist.x_val, mnist.y_val),
                                  steps_per_epoch=mnist.x_train.shape[0] // mnist.batch_size,
                                  callbacks=[mnist.lerning_rate_reduction])

mnist.evaluate()

y_pred = mnist.predict(mnist.x_test, classes=True)

df = pd.DataFrame({ 'ImageId': list(range(1, len(y_pred) + 1)), 'Label': y_pred })
df.to_csv('results.csv', sep=',', index=False, index_label='ImageId')