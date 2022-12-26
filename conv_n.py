import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Game:
    def __init__(self):
        self.turn=0
        self.table=[[-0.5, -0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                    [-0.5, -0.5, -0.5]]
    def control_game(seimport tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Game:
    def __init__(self):
        self.turn=0
        self.table=[[-0.5, -0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                    [-0.5, -0.5, -0.5]]
    def control_game(self):
        if self.table[:][0]==[1,1,1] or self.table[:][0]==[-2, -2, -2]:
            return False
        elif self.table[:][1]==[1,1,1] or self.table[:][1]==[-2, -2, -2]:
            return False
        elif self.table[:][2]==[1,1,1] or self.table[:][2]==[-2, -2, -2]:
            return False
        elif self.table[0][:]==[1,1,1] or self.table[0][:]==[-2, -2, -2]:
            return False
        elif self.table[1][:]==[1,1,1] or self.table[1][:]==[-2, -2, -2]:
            return False
        elif self.table[2][:]==[1,1,1] or self.table[2][:]==[-2, -2, -2]:
            return False
        elif [self.table[0][0],self.table[1][1], self.table[2][2]]==[1,1,1] or [self.table[0][0],self.table[1][1], self.table[2][2]]==[-2,-2,-2]:
            return False
        elif [self.table[0][2], self.table[1][1], self.table[2][0]]==[1,1,1] or [self.table[0][2], self.table[1][1], self.table[2][0]]==[-2, -2, -2]:
            return False
        return True
    def start_game(self, model):
        max_index_x=0
        max_index_y=0
        max_=-100
        while self.control_game():
            x,y=0,0
            d=np.array([x for x in self.table])
            d=d.astype(float)
            if self.turn==0:
                while y<3:
                    while x<3:
                        if d[y][x]==-0.5:
                            d[y][x]=1
                            val=model.predict(d.reshape(1,3,3))
                            if val>max_:
                                max_=val
                                max_index_y=y
                                max_index_x=x
                            d[y][x]=-0.5
                        x=x+1
                    x=0
                    y=y+1
                self.turn=1
                if max_!=-100:
                    self.table[max_index_y][max_index_x]=1
                max_=-100
            else:
                x=int(input())
                self.table[int(x/3)][x%3]=-2
                self.turn=0
            for x in self.table:
                for y in x:
                    if y==1:
                        print("x", end="")
                    elif y==-2:
                        print("o", end="")
                    else:
                        print("e", end="")
                print("")
data=pd.read_csv('tic-tac-toe_csv.csv')
data=data.to_numpy()
print(data.shape)

x=0
y=0
while y<len(data):
    while x<len(data[y]):
        if data[y][x]=='x':
            data[y][x]=1
        elif data[y][x]=='o':
            data[y][x]=-2
        elif data[y][x]=='b':
            data[y][x]=-0.5
        elif data[y][x]=='positive':
            data[y][x]=1
        else:
            data[y][x]=0
        x+=1
    y+=1
    x=0
data = np.unique(data.astype(float), axis=0)
print(data.shape)
print(data)
outputs=np.array([[x[-1]] for x in data]).reshape(len(data),1)
inputs=np.array([x[:-1] for x in data]).reshape(len(data),3,3)
X_train, X_test, y_train, y_test=train_test_split(inputs, outputs, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val=train_test_split(X_test, y_test, test_size=0.5, random_state=42)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
                                 input_shape=(3,3,1),
                                 kernel_size=(1,1),
                                 padding="same",
                                 filters=3,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2DTranspose(kernel_size=(7,7),
                                 
                                 filters=4,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(2,2),
                                 filters=5,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(
                                 kernel_size=(2,2),
                                 filters=6,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(1,1),
                                 padding="same",
                                 filters=7,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(1,1),
                                 filters=6,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation="tanh"))

model.compile(optimizer="adam", 
              loss="binary_crossentropy",
 metrics=['accuracy'])
model.summary()
inputs=inputs.astype(float)
outputs=outputs.astype(float)
results=model.fit(epochs=32, x=inputs, y=outputs)
print(model.predict(inputs[0].reshape(1, 3, 3)))
plt.plot(results.history['loss'], label="loss")
plt.plot(results.history['accuracy'], label="accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
Game().start_game(model)lf):
        if self.table[:][0]==[1,1,1] or self.table[:][0]==[-2, -2, -2]:
            return False
        elif self.table[:][1]==[1,1,1] or self.table[:][1]==[-2, -2, -2]:
            return False
        elif self.table[:][2]==[1,1,1] or self.table[:][2]==[-2, -2, -2]:
            return False
        elif self.table[0][:]==[1,1,1] or self.table[0][:]==[-2, -2, -2]:
            return False
        elif self.table[1][:]==[1,1,1] or self.table[1][:]==[-2, -2, -2]:
            return False
        elif self.table[2][:]==[1,1,1] or self.table[2][:]==[-2, -2, -2]:
            return False
        elif [self.table[0][0],self.table[1][1], self.table[2][2]]==[1,1,1] or [self.table[0][0],self.table[1][1], self.table[2][2]]==[-2,-2,-2]:
            return False
        elif [self.table[0][2], self.table[1][1], self.table[2][0]]==[1,1,1] or [self.table[0][2], self.table[1][1], self.table[2][0]]==[-2, -2, -2]:
            return False
        return True
    def start_game(self, model):
        max_index_x=0
        max_index_y=0
        max_=-100
        while self.control_game():
            x,y=0,0
            d=np.array([x for x in self.table])
            d=d.astype(float)
            if self.turn==0:
                while y<3:
                    while x<3:
                        if d[y][x]==-0.5:
                            d[y][x]=1
                            val=model.predict(d.reshape(1,3,3))
                            if val>max_:
                                max_=val
                                max_index_y=y
                                max_index_x=x
                            d[y][x]=-0.5
                        x=x+1
                    x=0
                    y=y+1
                self.turn=1
                if max_!=-100:
                    self.table[max_index_y][max_index_x]=1
                max_=-100
            else:
                x=int(input())
                self.table[int(x/3)][x%3]=-2
                self.turn=0
            for x in self.table:
                for y in x:
                    if y==1:
                        print("x", end="")
                    elif y==-2:
                        print("o", end="")
                    else:
                        print("e", end="")
                print("")
data=pd.read_csv('tic-tac-toe_csv.csv')
data=data.to_numpy()
print(data.shape)

x=0
y=0
while y<len(data):
    while x<len(data[y]):
        if data[y][x]=='x':
            data[y][x]=1
        elif data[y][x]=='o':
            data[y][x]=-2
        elif data[y][x]=='b':
            data[y][x]=-0.5
        elif data[y][x]=='positive':
            data[y][x]=1
        else:
            data[y][x]=0
        x+=1
    y+=1
    x=0
data = np.unique(data.astype(float), axis=0)
print(data.shape)
print(data)
outputs=np.array([[x[-1]] for x in data]).reshape(len(data),1)
inputs=np.array([x[:-1] for x in data]).reshape(len(data),3,3)
X_train, X_test, y_train, y_test=train_test_split(inputs, outputs, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val=train_test_split(X_test, y_test, test_size=0.5, random_state=42)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
                                 input_shape=(3,3,1),
                                 kernel_size=(1,1),
                                 padding="same",
                                 filters=3,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2DTranspose(kernel_size=(7,7),
                                 
                                 filters=4,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(2,2),
                                 filters=5,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(
                                 kernel_size=(2,2),
                                 filters=6,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(1,1),
                                 padding="same",
                                 filters=7,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(1,1),
                                 filters=6,
                                 activation="tanh",
                                 kernel_initializer="he_uniform"
                                 ))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation="tanh"))

model.compile(optimizer="adam", 
              loss="binary_crossentropy",
 metrics=['accuracy'])
model.summary()
inputs=inputs.astype(float)
outputs=outputs.astype(float)
results=model.fit(epochs=32, x=inputs, y=outputs)
print(model.predict(inputs[0].reshape(1, 3, 3)))
plt.plot(results.history['loss'], label="loss")
plt.plot(results.history['accuracy'], label="accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
Game().start_game(model)
