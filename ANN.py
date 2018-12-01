import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.metrics import confusion_matrix,accuracy_score
import timeit

start = timeit.default_timer()

dataset=pd.read_csv('Churn_Modelling.csv')
features=['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']
X=dataset[features]                  #X=dataset.iloc[:,3:13].values
Y=dataset['Exited']                  #Y=dataset.iloc[:,13].values

encoder=LabelEncoder()
X['Geography']=encoder.fit_transform(X['Geography'])
geography=X['Geography']
geography=pd.get_dummies(geography)
X=X.drop(['Geography'],axis=1)
X['Gender']=encoder.fit_transform(X['Gender'])
X['Geo1']=geography[0]
X['Geo2']=geography[1]
#print(X.head(100))

X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=1)

Scaler=StandardScaler()
X_Train=Scaler.fit_transform(X_Train)
X_Test=Scaler.fit_transform(X_Test)

                                                                        #SVM Classifier accuracy=0.86
from keras.models import Sequential                                     #from sklearn.svm import SVC
from keras.layers import Dense                                          #clf=SVC(C=1,kernel='rbf',gamma='scale')
                                                                        #clf.fit(X_Train,Y_Train)
                                                                        #predict=clf.predict(X_Test)
#clf=Sequential()                                                        #confidence=accuracy_score(Y_Test,predict)
                                                                        #print(confidence)

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
def build_classifier():
       clf=Sequential()
       clf.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))
       clf.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
       clf.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
       clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
       return clf

#clf.fit(X_Train,Y_Train,batch_size=10,epochs=100)   without CV
clf=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)

accuracy=cross_val_score(clf,X_Train,Y_Train,cv=10,n_jobs=-1)
print(accuracy.mean())


#predict=clf.predict(X_Test)
#predict= (predict>0.5)
#print(accuracy_score(Y_Test,predict))

stop = timeit.default_timer()

print('Time: ', stop - start)