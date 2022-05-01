from __future__ import print_function

import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from sklearn.svm import SVC
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
from pyspark.ml.feature import *
from pyspark.mllib import *
from pyspark.ml.classification import *
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
import pandas as pd
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import RandomForestClassifier
import matplotlib.pyplot as plt
from pyspark.ml.classification import GBTClassifier
if len(sys.argv) != 4:
        print("Usage: AUS <file> <output> ", file=sys.stderr)
        exit(-1)
l = [] # output list
#df = sc.textFile(sys.argv[1])

df = spark.read.csv(sys.argv[1], header = True, inferSchema = True)
df = df.drop('customerID')
df = df.withColumn("InternetService", F.when(F.col("InternetService") == 2,1).otherwise(F.col("InternetService")))
### replacing OnlineSecurity, Streaming TV, Streaming Movies,, Devou ce Protection, Techsupport as 0 when InternetService is 0, as we could NOT get these sevices from a telecom
## provider when you do not have Interet service from the provider 
df = df.withColumn("OnlineSecurity", F.when(F.col("InternetService") == 0,0).otherwise(F.col("OnlineSecurity")))
df = df.withColumn("DeviceProtection", F.when(F.col("InternetService") == 0,0).otherwise(F.col("DeviceProtection")))
df = df.withColumn("TechSupport", F.when(F.col("InternetService") == 0,0).otherwise(F.col("TechSupport")))
df = df.withColumn("StreamingTV", F.when(F.col("InternetService") == 0,0).otherwise(F.col("StreamingTV")))
df = df.withColumn("StreamingMovies", F.when(F.col("InternetService") == 0,0).otherwise(F.col("StreamingMovies")))

#  Converting the values into wether or not there is a Internet service, and payment into autopay or non autopay
# and contract variable in to wether or not a customer has a contract with the service provider
df = df.withColumn("InternetService", F.when(F.col("InternetService") == 2,1).otherwise(F.col("InternetService")))
df = df.withColumn("Contract", F.when(F.col("Contract") == 2,1).otherwise(F.col("Contract")))
df = df.withColumn("PaymentMethod", F.when(F.col("PaymentMethod") == 2,1).otherwise(F.col("PaymentMethod")))
df = df.withColumn("PaymentMethod", F.when(F.col("PaymentMethod") == 3,1).otherwise(F.col("PaymentMethod")))
# and replacing the 2 value for online ecurity ,device protection, techsupport, streaming tv as Null to remove them (as they are nulls in data)

df = df.withColumn("OnlineSecurity", F.when(F.col("OnlineSecurity") == 2,None).otherwise(F.col("OnlineSecurity")))
df = df.withColumn("DeviceProtection", F.when(F.col("DeviceProtection") == 2,None).otherwise(F.col("DeviceProtection")))
df = df.withColumn("TechSupport", F.when(F.col("TechSupport") == 2,None).otherwise(F.col("TechSupport")))
df = df.withColumn("StreamingTV", F.when(F.col("StreamingTV") == 2,None).otherwise(F.col("StreamingTV")))
df = df.withColumn("StreamingMovies", F.when(F.col("StreamingMovies") == 2,None).otherwise(F.col("StreamingMovies")))
df = df.na.drop("any")

df = df.withColumnRenamed("churn", "label")

### checking the weights of churn column, generally the Churned customer weight is relatively low comparing to not churned so balancing the data set
churn_Weight = df.filter('label==1').count()/df.count()
NotChurn_weight = df.filter('label==0').count()/df.count()
print(churn_Weight,NotChurn_weight)
df1 = df.withColumn("weight",F.when(F.col("label")==1,NotChurn_weight).otherwise(churn_Weight))
pd.DataFrame(df1.take(5), columns=df1.columns)

######### Transform the vector assembler to get the vector of feature input columns

assembler = VectorAssembler(
    inputCols=["gender", "SeniorCitizen", "Partner", "Dependents","Tenure","PhoneService","MultipleLines","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"],
    outputCol="features")

df1 = assembler.transform(df1)
pd.DataFrame(df1.take(5), columns=df1.columns)



###### logistic regression Classifier ###################
classifier = LogisticRegression(maxIter=20,weightCol="weight",threshold= 0.6,family='auto')
model_log1 = classifier.fit(df1)
#### Transforming on traiing data
prediction_train = model_log1.transform(df1)
prediction_train = prediction_train.select("label","prediction").rdd.map(lambda x: (float(x[0]),float(x[1]),(1 if float(x[0])==1 and float(x[1])==1 else 0),
                 (1 if float(x[0])==0 and float(x[1])==1 else 0),(1 if float(x[0])==1 and float(x[1])==0 else 0),(1 if float(x[0])==0 and float(x[1])==0 else 0),))
prediction_train.take(10)

prediction_train = prediction_train.map(lambda x: (float(x[2]),float(x[3]),float(x[4]),float(x[5])))\
                            .reduce(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3]))

## Training Confusion Matrix
print("Confusion Matrix Training")
print(np.reshape(prediction_train,(2,2)))

precision_train = prediction_train[0]/(prediction_train[0]+prediction_train[1])
recall_train = prediction_train[0]/(prediction_train[0]+prediction_train[2])
Fmeasure_log_train = (2*precision_train*recall_train)/(precision_train+recall_train)

print("Precision Logistic Train: ",precision_train,"Recall SVC: ", recall_train,"Fmeasure SVC:  ", Fmeasure_log_train)

l.append("Precision,Recall, Fmeasure of Training Logistic Data")
l.append([precision_train,recall_train,Fmeasure_log_train])

### Fitting model on SVC
svc_classifier = LinearSVC(maxIter=20,weightCol="weight")
model_svc = svc_classifier.fit(df1)




###################### Test Data ##########################################################

df_test = spark.read.csv(sys.argv[2], header = True, inferSchema = True)
df_test = df_test.drop('customerID')
df_test = df_test.withColumn("InternetService", F.when(F.col("InternetService") == 2,1).otherwise(F.col("InternetService")))

# Data transformation: Converting the values into wether or not there is a Internet service, and payment into autopay or non autopay
# and contract variable in to wether or not a customer has a contract with the service provider
# and replacing the 2 value for online ecurity ,device protection, techsupport, streaming tv as Null to remove them (as they are nulls in data)

df_test = df_test.withColumn("InternetService", F.when(F.col("InternetService") == 2,1).otherwise(F.col("InternetService")))

df_test = df_test.withColumn("OnlineSecurity", F.when(F.col("InternetService") == 0,0).otherwise(F.col("OnlineSecurity")))
df_test = df_test.withColumn("DeviceProtection", F.when(F.col("InternetService") == 0,0).otherwise(F.col("DeviceProtection")))
df_test = df_test.withColumn("TechSupport", F.when(F.col("InternetService") == 0,0).otherwise(F.col("TechSupport")))
df_test = df_test.withColumn("StreamingTV", F.when(F.col("InternetService") == 0,0).otherwise(F.col("StreamingTV")))
df_test = df_test.withColumn("StreamingMovies", F.when(F.col("InternetService") == 0,0).otherwise(F.col("StreamingMovies")))

df_test = df_test.withColumn("InternetService", F.when(F.col("InternetService") == 2,1).otherwise(F.col("InternetService")))
df_test = df_test.withColumn("Contract", F.when(F.col("Contract") == 2,1).otherwise(F.col("Contract")))

df_test = df_test.withColumn("PaymentMethod", F.when(F.col("PaymentMethod") == 2,1).otherwise(F.col("PaymentMethod")))
df_test = df_test.withColumn("PaymentMethod", F.when(F.col("PaymentMethod") == 3,1).otherwise(F.col("PaymentMethod")))

df_test = df_test.withColumn("OnlineSecurity", F.when(F.col("OnlineSecurity") == 2,None).otherwise(F.col("OnlineSecurity")))
df_test = df_test.withColumn("DeviceProtection", F.when(F.col("DeviceProtection") == 2,None).otherwise(F.col("DeviceProtection")))
df_test = df_test.withColumn("TechSupport", F.when(F.col("TechSupport") == 2,None).otherwise(F.col("TechSupport")))
df_test = df_test.withColumn("StreamingTV", F.when(F.col("StreamingTV") == 2,None).otherwise(F.col("StreamingTV")))
df_test = df_test.withColumn("StreamingMovies", F.when(F.col("StreamingMovies") == 2,None).otherwise(F.col("StreamingMovies")))
df_test = df_test.na.drop("any")


df_test = df_test.withColumnRenamed("churn", "label")
churn_Weight = df_test.filter('label==1').count()/df_test.count()
NotChurn_weight = df_test.filter('label==0').count()/df_test.count()
print(churn_Weight,NotChurn_weight)
df_test1 = df_test.withColumn("weight",F.when(F.col("label")==1,NotChurn_weight).otherwise(churn_Weight))
pd.DataFrame(df_test1.take(5), columns=df_test1.columns)



assembler = VectorAssembler(
    inputCols=["gender", "SeniorCitizen", "Partner", "Dependents","Tenure","PhoneService","MultipleLines","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"],
    outputCol="features")


df_test1 = assembler.transform(df_test1)
pd.DataFrame(df_test1.take(5), columns=df_test1.columns)

###### transform Test data with what is fit in Training data ####
prediction_test = model_log1.transform(df_test1)
prediction_test = prediction_test.select("label","prediction").rdd.map(lambda x: (float(x[0]),float(x[1]),(1 if float(x[0])==1 and float(x[1])==1 else 0),
                 (1 if float(x[0])==0 and float(x[1])==1 else 0),(1 if float(x[0])==1 and float(x[1])==0 else 0),(1 if float(x[0])==0 and float(x[1])==0 else 0),))
prediction_test.take(10)

prediction_test = prediction_test.map(lambda x: (float(x[2]),float(x[3]),float(x[4]),float(x[5])))\
                            .reduce(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3]))

print("Confusion Matrix Test Data Logistic")
print(np.reshape(prediction_test,(2,2)))


################# Logistic MODEL #####################
print ("################# Logistic MODEL #####################")
print("Confusion Matrix Training")
print(np.reshape(prediction_test,(2,2)))

precision_test = prediction_test[0]/(prediction_test[0]+prediction_test[1])
recall_test = prediction_test[0]/(prediction_test[0]+prediction_test[2])
Fmeasure_log_test = (2*precision_test*recall_test)/(precision_test+recall_test)

print("Precision Logistic Test: ",precision_test,"Recall Logisctic Test: ", recall_test,"Fmeasure :  ", Fmeasure_log_test)

l.append("Precision,Recall, Fmeasure of Logistic Testing Data")
l.append([precision_test,recall_test,Fmeasure_log_test])
l.append("Confusion Matrix")
l.append(np.reshape(prediction_test,(2,2)))
################# SVC MODEL #####################
svc_classifier = LinearSVC(maxIter=10,weightCol='weight')
model_svc = svc_classifier.fit(df1)

### Transform on testing data
svc_output = model_svc.transform(df_test1)
prediction_svctest = svc_output.select("label","prediction").rdd.map(lambda x: (float(x[0]),float(x[1])))
svc_stats = MulticlassMetrics(prediction_svctest)

l.append("######## SVC precision, recall and fmeasure  ###########################")
l.append([svc_stats.precision(1.0),svc_stats.recall(1.0),svc_stats.fMeasure(1.0)])
l.append("Confusion Matrix")
l.append(svc_stats.confusionMatrix())

##########  Random Forest ############

ranfor_classifier = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="label", seed=777,weightCol="weight",impurity='entropy',maxBins=32)
model_ranfor = ranfor_classifier.fit(df1) # fit on training data

### Transform on testing data
ranfor_output = model_ranfor.transform(df_test1)
prediction_ranfortest = ranfor_output.select("label","prediction").rdd.map(lambda x: (float(x[0]),float(x[1])))
ranfor_stats = MulticlassMetrics(prediction_ranfortest)

l.append("######## RANDOM FOREST precision, recall and fmeasure  ###########################")
l.append([ranfor_stats.precision(1.0),ranfor_stats.recall(1.0),ranfor_stats.fMeasure(1.0)])
l.append("Confusion Matrix")
l.append(ranfor_stats.confusionMatrix())


######################### GBT #######

gbt_classifier = gbt_classifier = GBTClassifier(maxIter=20,weightCol="weight",stepSize=0.3, minInfoGain=0.2)
model_gbt = gbt_classifier.fit(df1)
### Transform on testing data
gbt_output = model_gbt.transform(df_test1)
prediction_gbttest = gbt_output.select("label","prediction").rdd.map(lambda x: (float(x[0]),float(x[1])))
gbt_stats = MulticlassMetrics(prediction_gbttest)

l.append("######## Gradient Boosting Tree precision, recall and fmeasure  ###########################")
l.append([gbt_stats.precision(1.0),gbt_stats.recall(1.0),gbt_stats.fMeasure(1.0)])
l.append("Confusion Matrix")
l.append(gbt_stats.confusionMatrix())

################## Naive Bayes #####################################

NB_classifier  = NaiveBayes(modelType="multinomial",weightCol="weight",smoothing=1.0)
model_NB = NB_classifier.fit(df1)

### Transform on testing data
NB_output = model_NB.transform(df_test1)
prediction_NBtest = NB_output.select("label","prediction").rdd.map(lambda x: (float(x[0]),float(x[1])))
NB_stats = MulticlassMetrics(prediction_NBtest)

l.append("######## Naive Bayes precision, recall and fmeasure  ###########################")
l.append([NB_stats.precision(1.0),NB_stats.recall(1.0),NB_stats.fMeasure(1.0)])
l.append("Confusion Matrix")
l.append(NB_stats.confusionMatrix())


############## OUTPUT #######################
All_CM = sc.parallelize(l).coalesce(1)
All_CM.saveAsTextFile(sys.argv[3])
sc.stop()