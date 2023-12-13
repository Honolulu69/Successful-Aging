#Initial import of Libraries

import pandas as pd # used to load, manipulate the data and for one-hot encoding
import numpy as np # data manipulation
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.model_selection import train_test_split # for splitting the dataset into train and test split
from sklearn.preprocessing import scale # scale and center the data
from sklearn.svm import SVC # will make a SVM for classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns

#Importing the dataset
df = pd.read_csv('C:/Users/A/Desktop/SAnet/SAnettrue/SAData1.csv')

pd.set_option('display.max_columns', None) # will show the all columns with pandas dataframe
pd.set_option('display.max_rows', None) # will show the all rows with pandas dataframe

df.head()

df.shape

df.info()

#Data Preprocessing
df['Hypertension/CVD'] = df['Hypertension/CVD'].replace(['Present','Absent'], [1,0]) # Hyp column
df['Renal Disease'] = df['Renal Disease'].replace(['Yes','No'], [1,0]) # Renal column
df['Liver Disease'] = df['Liver Disease'].replace(['Yes','No'], [1,0]) # Liver column
df['NMD'] = df['NMD'].replace(['Yes','No'], [1,0]) # NMD column
df['Depression'] = df['Depression'].replace(['Yes','No'], [1,0]) # Depression column
df['Eye disease'] = df['Eye disease'].replace(['Yes','No'], [1,0]) # Depression column
df['Diabetes'] = df['Diabetes'].replace(['Yes','No'], [1,0]) # Depression column
df['Cancer'] = df['Cancer'].replace(['Present','Absent'], [1,0]) # Depression column
df['ADLs'] = df['ADLs'].replace(['Inability','Ability'], [1,0]) # Depression column
df['Nutrition Status'] = df['Nutrition Status'].replace(['Nourished','Malnourished'], [1,0]) # Depression column
df['QOL'] = df['QOL'].replace(['Abnormal','Standard'], [1,0]) # Depression column


from sklearn.preprocessing import LabelEncoder

correlation_matrix = df.corr()
data_corr = correlation_matrix['CLASS'].sort_values(ascending=False)
data_corr

df.isnull().sum()

y = df['Class'].values
X = df[['Age', 'Hypertension/CVD', 'Renal Disease', 'Liver Disease', 'NMD', 'Depression', 'Eye disease', 
        'Diabetes', 'Cancer', 'ADLs', 'Nutrition Status', 'QOL']]

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Data Split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size= 0.50, random_state=0, stratify=y)

df_ytrain = pd.DataFrame(y_trainval)
df_ytest = pd.DataFrame(y_test)

print('In Training Split:')
print(df_ytrain[0].value_counts())

print('\nIn Testing Split:')
print(df_ytest[0].value_counts())

#Data Scaling
# here StandardScaler() means z = (x - u) / s
scaler = StandardScaler().fit(X_trainval)

#scaler = MinMaxScaler().fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

X_trainval_scaled

X_trainval.describe()

#ANN model for Successful Aging
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, AlphaDropout
from keras.callbacks import ModelCheckpoint

#Initializing the neural network classifier
classideep = Sequential()

#Adding the input layer and the first hidden layer and the drop out nodes
classideep.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu', input_dim = 12))
classideep.add(Dropout(0.20))

#Adding the second hidden layer and the drop out nodes
classideep.add(Dense(units = 3, kernel_initializer = 'random_uniform', activation = 'relu'))
classideep.add(Dropout(0.20))

#Adding the output layer of the neural neywork classifier
classideep.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = 'sigmoid'))

classideep.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = classideep.fit(X_trainval, y_trainval, validation_split = 0.2, batch_size = 32, epochs = 500)

scores = classideep.evaluate(X_trainval, y_trainval, verbose = 0)
print("%s: %.2f%%" % (classideep.metrics_names[1], scores[1]*100))

scores = classideep.evaluate(X_test, y_test, verbose = 0)
print("%s: %.2f%%" % (classideep.metrics_names[1], scores[1]*100))


#Model Architecture
classideep.save("classipark.h5")

classideep = load_model('classipark.h5')
classideep.summary()

#ANN model prediction
y_pred = classideep.predict(X_test)
y_pred = (y_pred > 0.7)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt = "d", cmap="terrain_r")

print(history.history.keys())

#Data Visualization 
#ANN Model Accuracy Plot
plt.plot(history.history['accuracy'], color = 'navy')
plt.plot(history.history['val_accuracy'], color = 'violet')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='lower left')
plt.show()

#ANN Model Loss Plot
plt.plot(history.history['loss'], color = 'navy')
plt.plot(history.history['val_loss'], color = 'violet')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='lower left')
plt.show()


#Support Vector Machine Model
# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC( kernel = 'rbf' , random_state = 0)
classifier.fit(X_trainval, y_trainval)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)
sns.heatmap(cm, annot = True, fmt = "d", cmap="RdYlBu_r")


#Naive Bayes Model
# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_trainval, y_trainval)

# Predicting the Test set results
y_pred2 = classifier.predict(X_test)

import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
sns.heatmap(cm, fmt = "d",annot = True, cmap = 'icefire')


