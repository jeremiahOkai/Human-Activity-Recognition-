from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.layers import LSTM,Dropout, Activation, GRU, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from Generator_v10 import Generator
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers
from keras.regularizers import l2, l1

class Model: 
    
    def __init__(self, 
                 name, 
                 X_train,
                 y_train,
                 X_val, y_val, 
                 n_batch = 32, n_epochs=200): 
        self.n_batch = n_batch
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.name = name 
        self.n_epochs = n_epochs
#         self.Generator = Generator
        
    def Dense_basemodel(self):
        print('Creating model')
        model = Sequential()
        model.add(GRU(128, return_sequences=False, kernel_regularizer=l2(l=0.01), recurrent_dropout=0.4, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dense(256, kernel_regularizer=l2(l=0.01)))
        model.add(Dense(512, kernel_regularizer=l2(l=0.01))) 
        model.add(Dense(256, kernel_regularizer=l2(l=0.01)))
        model.add(Dense(16, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize256_dropoutratio50', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
        return history 
    
    def GRU_basemodel(self):
        print('Creating model')
        model = Sequential()
        model.add(GRU(64,return_sequences=True, dropout=0.4, recurrent_dropout=0.5, 
                      input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(GRU(128, return_sequences=True, dropout=0.6, recurrent_dropout=0.5)) 
        model.add(GRU(64, return_sequences=True ,  dropout=0.5, recurrent_dropout=0.5)) 
        model.add(GRU(32, return_sequences=False, dropout=0.4, recurrent_dropout=0.5))
        model.add(Dense(16, activation='softmax'))
        opt = Adam(epsilon=1e-08)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize256_dropoutratio50', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
    
    
    def LSTM_basemodel(self):
        print('Creating model')
        model = Sequential()
        model.add(LSTM(64, return_sequences=True,  dropout=0.0,kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5, 
                      input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(LSTM(128, return_sequences=True,  dropout=0.2, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5)) 
        model.add(LSTM(64, return_sequences=True ,  dropout=0.2,kernel_regularizer=l2(l=0.001),  recurrent_dropout=0.5)) 
        model.add(LSTM(32, return_sequences=False,  dropout=0.2, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5))
        model.add(Dense(16, activation='softmax'))
        opt=Adam(epsilon=1e-08)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize256_dropoutratio50', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
        return history
    
    
    def model_deep(self):
        print('Creating model')
        model = Sequential()
        model.add(GRU(64, return_sequences=True,  dropout=0.4, recurrent_dropout=0.5, 
                      input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(GRU(128, return_sequences=True, dropout=0.6, recurrent_dropout=0.5)) 
        model.add(GRU(256, return_sequences=True,  dropout=0.6, recurrent_dropout=0.5))
        model.add(GRU(128, return_sequences=True, dropout=0.6, recurrent_dropout=0.5))
        model.add(GRU(64, return_sequences=True ,  dropout=0.4, recurrent_dropout=0.5)) 
        model.add(GRU(32, return_sequences=False, dropout=0.4, recurrent_dropout=0.5))
        model.add(Dense(16, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize256_dropoutratio50', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
        return history 
    
    def try_1(self):
        print('Creating model')
        model = Sequential()
        model.add(GRU(32,return_sequences=True, dropout=0.0, recurrent_dropout=0.2, 
                      input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(GRU(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
        model.add(GRU(32, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dense(16, activation='softmax'))
        opt = Adam(epsilon=1e-08)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize256_dropoutratio50', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
        return history
    
    def try_2(self):
        print('Creating model')
        model = Sequential()
        model.add(GRU(32,return_sequences=True, dropout=0.2, recurrent_dropout=0.2, 
                      input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(GRU(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(16, activation='softmax'))
        opt = Adam(epsilon=1e-08)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize128_dropoutratio100', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
        return history
    
    def try_3(self):
        print('Creating model')
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(LSTM(512, return_sequences=True, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5)) 
        model.add(LSTM(512, return_sequences=False, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5))
        model.add(Dense(16, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize32_dropoutratio100', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
        return history
    
    def try_4(self):
        print('Creating model')
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(LSTM(512, return_sequences=True, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5)) 
        model.add(LSTM(512, return_sequences=False, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5))
        model.add(Dense(16, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize32_dropoutratio100', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
        return history

    def try_5(self):
        print('Creating model')
        model = Sequential()
        model.add(GRU(512, return_sequences=True, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(GRU(512, return_sequences=True, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5)) 
        model.add(GRU(512, return_sequences=False, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5))
        model.add(Dense(16, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize32_dropoutratio100', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout = 2, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout = 2, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
        return history
    
    def try_6(self):
        print('Creating model')
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(LSTM(512, return_sequences=True, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5)) 
        model.add(LSTM(512, return_sequences=False, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5))
        model.add(Dense(16, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize32_dropoutratio100', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
        return history
    
    def try_7(self):
        print('Creating model')
        model = Sequential()
        model.add(GRU(512, return_sequences=True, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(GRU(512, return_sequences=True, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5)) 
        model.add(GRU(512, return_sequences=False, kernel_regularizer=l2(l=0.001), recurrent_dropout=0.5))
        model.add(Dense(16, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='/home/s1931628/zeros2_model/'+ self.name+'_.hdf5', save_best_only=True, verbose=1, mode='auto', monitor='val_loss')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, mode='auto', verbose=1)
        tbCallBack = TensorBoard(log_dir='/home/s1931628/Graph/'+self.name+'/GRUbatchsize32_dropoutratio100', histogram_freq=0, write_graph=True, write_images=True)
        my_generator = Generator(self.X_train, self.y_train, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0)
        val_generator = Generator(self.X_val, self.y_val, batchsize=self.n_batch, dropout_ratio=1.0, noisedrop_ratio = 0.0,  noisedrop=4, downsample=False)
        history = model.fit_generator(my_generator.generate(), steps_per_epoch=int(my_generator.Xtrain_balanced_size/self.n_batch )+1,
                                  epochs=self.n_epochs, verbose=1, 
                                  validation_data=val_generator.generate(), validation_steps=1, 
                                  shuffle=False, callbacks=[checkpointer,tbCallBack])
        return history
        
    
   