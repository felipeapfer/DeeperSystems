import numpy
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import *
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.preprocessing import MinMaxScaler

from keras import backend as K


train = pd.read_csv('./Data/Train/train_100k.csv',index_col='id')
train_y = pd.read_csv('./Data/Train/train_100k.truth.csv',index_col='id')
test = pd.read_csv('./Data/Test/test_100k.csv',index_col='id')


scalar = MinMaxScaler()
split_size = int(train.shape[0]*0.7)
train_x, val_x = train[:split_size], train[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]
scalar.fit(train_x)
x=scalar.transform(train_x)
val_x=scalar.transform(val_x)




def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def build_model(learning_rate,activation,drop_rate,num_unit):
    keras.backend.clear_session()
    model=Sequential()
    model.add(Dense(num_unit,activation=activation,kernel_initializer='normal',input_dim=train.shape[1]))
    model.add(Dropout(drop_rate))
    model.add(Dense(num_unit,kernel_initializer='normal',activation=activation))
    model.add(Dropout(drop_rate))
    model.add(Dense(2,kernel_initializer='normal'))
    model.compile(loss=root_mean_squared_error ,optimizer=keras.optimizers.Adam(lr=learning_rate))
    return model



batch_size=[5000]
epochs = [250,500,1000]
learning_rate = [0.01,0.001,0.1]
drop_rate = [0.4,0.30,0.20,0.15]
num_unit = [200,150,100,80,60]
activation= ['tanh','relu','linear']

parameters = dict(batch_size=batch_size,
                  epochs=epochs,
                  drop_rate=drop_rate,
                  learning_rate=learning_rate,
                  num_unit=num_unit,
                  activation=activation)
m_validated=KerasRegressor(build_fn=build_model,verbose=1)
models = GridSearchCV(estimator=m_validated, param_grid=parameters,n_jobs=1)

best_model=models.fit(x,train_y,validation_data=(val_x, val_y))
print("Best Model")
print(best_model.best_params_)

test_scaled=scalar.transform(test)
out = best_model.predict(test_scaled)

file = open('env/submission.csv','w') 
file.write('id,slope,intercept\n')
for i in range(len(out)):
    file.write(str(i)+','+str(out[i][0])+','+str(out[i][1])+'\n')
file.close()