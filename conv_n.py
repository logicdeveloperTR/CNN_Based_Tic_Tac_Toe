import tensorflow as tf
import numpy as np
import pandas as pd

class Game:
    def __init__(self):
        self.turn=0
        self.table=[[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]
    def control_game(self):
        if self.table[:][0]==[1,1,1] or self.table[:][0]==[-1, -1, -1]:
            return False
        elif self.table[:][1]==[1,1,1] or self.table[:][1]==[-1, -1, -1]:
            return False
        elif self.table[:][2]==[1,1,1] or self.table[:][2]==[-1, -1, -1]:
            return False
        elif self.table[0][:]==[1,1,1] or self.table[0][:]==[-1, -1, -1]:
            return False
        elif self.table[1][:]==[1,1,1] or self.table[1][:]==[-1, -1, -1]:
            return False
        elif self.table[2][:]==[1,1,1] or self.table[2][:]==[-1, -1, -1]:
            return False
        elif [self.table[0][0],self.table[1][1], self.table[2][2]]==[1,1,1] or [self.table[0][0],self.table[1][1], self.table[2][2]]==[1,1,1]:
            return False
        elif [self.table[0][2], self.table[1][1], self.table[2][0]]==[1,1,1] or [self.table[0][2], self.table[1][1], self.table[2][0]]==[-1, -1, -1]:
            return False
        return True
    def start_game(self, model):
        max_index_x=-1
        max_index_y=-1
        max_=0
        while self.control_game():
            x=0
            y=0
            d=np.array([x for x in self.table])
            d.astype(float)
            if self.turn==0:
                while y<3:
                    while x<3:
                        if d[y][x]==0:
                            d[y][x]=1
                            val=model.predict(d.reshape(1,3,3))
                            if val>max_:
                                max_=val
                                max_index_y=y
                                max_index_x=x
                            d[y][x]=0
                        x=x+1
                    x=0
                    y=y+1
                self.turn=1
                self.table[max_index_y][max_index_x]=1
                max_=0
            else:
                x=int(input())
                self.table[int(x/3)][x%3]=-1
                self.turn=0
            for x in self.table:
                print(x)
            
data=pd.read_csv('tic-tac-toe-endgame.csv')
data=data.to_numpy()
print(data)

x=0
y=0
while y<len(data):
    while x<len(data[y]):
        if data[y][x]=='x':
            data[y][x]=1
        elif data[y][x]=='o':
            data[y][x]=-1
        elif data[y][x]=='b':
            data[y][x]=0
        elif data[y][x]=='positive':
            data[y][x]=1
        else:
            data[y][x]=0
        x+=1
    y+=1
    x=0
print(data)
outputs=np.array([[x[-1]] for x in data]).reshape(len(data),1)
inputs=np.array([x[:-1] for x in data]).reshape(len(data),3,3)
print("d")
print(outputs)
print(inputs)
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
                                 input_shape=(3,3,1),
                                 kernel_size=(2,2),
                                 padding="same",
                                 filters=100,
                                 activation="relu",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(1,2),
                                 padding="same",
                                 filters=250,
                                 activation="relu",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(1,1),
                                 padding="same",
                                 filters=1000,
                                 activation="relu",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(1,1),
                                 padding="same",
                                 filters=750,
                                 activation="relu",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(1,2),
                                 padding="same",
                                 filters=500,
                                 activation="relu",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(2,2),
                                 padding="same",
                                 filters=250,
                                 activation="relu",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Conv2D(kernel_size=(1,1),
                                 padding="same",
                                 filters=100,
                                 activation="relu",
                                 kernel_initializer="he_uniform"
                                 ))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1000, activation="relu"))
model.add(tf.keras.layers.Dense(500, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", 
              loss="binary_crossentropy",
 metrics=['accuracy'])
inputs=inputs.astype(float)
outputs=outputs.astype(float)
model.fit(epochs=6, x=inputs, y=outputs)
Game().start_game(model)


                                