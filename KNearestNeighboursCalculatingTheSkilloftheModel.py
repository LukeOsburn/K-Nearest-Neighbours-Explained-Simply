import numpy as np
from sklearn import cross_validation
import pandas as pd
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter

#Lets predict the class for the test data and compare with actual and calculate the percentage of succcesful predictions

#lets read our data into a nice dataframe
df=pd.read_csv("breastdata.txt")

#lets replace missing values with a more recognised value
df=df.replace('?',-99999) #many alogirthims recognise -99999 as an outlier
#lets drop id, it is of no value
df=df.drop(['id'],1)


#lets do a simple 2 dimensional example

#Full=np.array(df.drop(['clump_thickness','unif_cell_size','class','unif_cell_shape','marg_adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','norm_nucleoli','mitoses'],1))
#drop all except 2, any 2, so we can visualise the algorithim
#lets turn into numpy arrays
X = np.array(df.drop(['class','unif_cell_shape','marg_adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','norm_nucleoli','mitoses'],1))

#this is our output, benign or malignant
#lets turn into numpy arrays
y = np.array(df['class'])

#randomly split the data into a training and test sets
#we will use the training test to "predict" benign or malignant from the test set
X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2)

#lets only take 20 data points for training, so we can visualise the results more easily
#X_train=X_train[0:40]
#y_train= y_train[0:40]

k=3 #number of neighbours

#okay, lets test all data points in our test conditions and calculate the success of the model
successes=0.0
fails=0.0
count=-1
for i in X_test:
    distances=[]
    count=count+1
    count2=-1
    for ii in X_train:
        count2=count2+1
        #this calculated the distance to each point in the training set in turn
        dist=sqrt(((i[0]-ii[0])**2 +(i[1]-ii[1])**2))
        #this is a list of lists of distances to each point and benign/malignant class
        distances.append([float(dist),float(y_train[count2])])


    #this sorts our list of lists by distance while preserving the benign/malignant class

    dist=sorted(distances, key=itemgetter(0))
    sortedT=map(list, zip(*dist))

    #this selects the first k members with the shortest distances
    Votes=sortedT[1][0:k]

    #this prints the most common class (benign/malignant and the number)
    #print(Counter(Votes).most_common(1))
    print("Data Point")
    print(i)
    print("Actual value")
    print(y_test[count])
    print("Predicted value")
    #this returns the most common class (benign/malignant and the number)
    predicted=Counter(Votes).most_common(1)[0][0]
    print(predicted)
    if predicted==y_test[count]:
        successes=successes+1
        print("Success")
    else:
        fails=fails+1
        print("Fail")



print(y_test)
skill=(round((successes/(successes+fails)),2))*100
print(successes/(successes+fails))

print("%s percent of the tumors were predicted correctly" %(skill))
