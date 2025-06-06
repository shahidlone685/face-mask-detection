from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Define the model
mymodel=Sequential()
mymodel.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3),activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3),activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Flatten())
mymodel.add(Dense(100,activation='relu'))
mymodel.add(Dense(1,activation='sigmoid'))
#compile model
mymodel.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
#Define the model
train=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test=ImageDataGenerator(rescale=1./255)
train_img=train.flow_from_directory('train',target_size=(150,150),batch_size=16,class_mode='binary')
test_img=test.flow_from_directory('test',target_size=(150,150),batch_size=16,class_mode='binary')
#Train and test the model
mask_model=mymodel.fit(train_img,epochs=10,validation_data=test_img)
#save the model in my directory
mymodel.save('mask.h5') 
