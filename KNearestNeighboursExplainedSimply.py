import numpy as np
from sklearn import cross_validation
import pandas as pd
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter

print("this is a simple script to demonstrates the use of k-nearest neighbors")
print("predicts the  class for a single data point and plots and demonstrates the result")


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
X_train=X_train[0:40]
y_train= y_train[0:40]


k=3 #number of neighbours

#okay, lets calculate the k nearest neighbours for one data point and plot it
for i in X_test[0:1]:
    distances=[]
    count=-1
    print(i)
    for ii in X_train:
        count=count+1
        #this calculated the distance to each point in the training set in turn
        dist=sqrt(((i[0]-ii[0])**2 +(i[1]-ii[1])**2))
        #this is a list of lists of distances to each point and benign/malignant class
        distances.append([float(dist),float(y_train[count])])


    #this sorts our list of lists by distance while preserving the benign/malignant class
    dist=sorted(distances, key=itemgetter(0))
    sortedT=map(list, zip(*dist))

    #this selects the first k members with the shortest distances
    Votes=sortedT[1][0:k]
    print("Votes")
    #this prints the most common class (benign/malignant and the number)
    print(Counter(Votes).most_common(1))




#this is just manipulating the data so we can plot it nicely and easily
#sorting training data into benign and malignant so we can plot it by color
XmaligX=[]
XmaligY=[]
XbenignX=[]
XbenignY=[]

counter=-1
for j in X_train:
    counter=counter+1
    if y_train[counter]==4:
        XmaligX.append(j[0])
        XmaligY.append(j[1])
    if y_train[counter]==2:
        XbenignX.append(j[0])
        XbenignY.append(j[1])


#lets print the predicted class in the title
predictedclass=Counter(Votes).most_common(1)[0][0]
if predictedclass==4:
    predictedclass='Malignant'
else:
    predictedclass='Benign'

#lets plot it
plt.scatter(XmaligX,XmaligY, color='r',label='Malignant')
plt.scatter(XbenignX,XbenignY, color='b',label='Benign')
plt.scatter(i[0],i[1],color='black',s=60,label='Tested Data Point')
plt.xlabel('Clump Thickness', fontsize=14, color='black')
plt.ylabel('Unif Cell Size', fontsize=14, color='black')
plt.title('Predict Class for 1 Data Point: %s' %(predictedclass), fontsize=16, color='black')
plt.legend(loc='upper left')
plt.show()
