#REF:https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/binary_classification_metrics_example.py
from pyspark import SparkConf, SparkContext
import numpy as np
from time import time
import os
import datetime
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionModel#TEST
from pyspark.mllib.evaluation import MulticlassMetrics

def CreateSparkContext():
    sparkConf = SparkConf()                                            \
                         .setAppName("RunDecisionTreeRegression")           \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
#    SetPath(sc)
    return (sc)

def SetLogger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

'''
def SetPath(sc):
    global Path
    if sc.master[0:5]=="local" :
        Path="file:/home/cs/Demo/"
    else:   
        Path="hdfs://master:9000/data/"
'''

#unuse
def convert_float(x):
	return float(x)

def LoadAndPrepare(dataset):
	f = sc.textFile(dataset).map(lambda x: str(x).replace('NULL','0'))
	header = f.first()
	f = f.filter(lambda x: x != header).map(lambda x:x.split(","))
	return f
	
def SaveOrLoadModel(model , path = ("model_" + str(datetime.date.today())) , option = 's'):
	if (option == 's'):
		model.save(sc, path)
	elif (option == 'l'):
		model = LogisticRegressionModel.load(sc, path)
	#model = LogisticRegressionModel(weights=[...], intercept=..., numFeatures = ... , numClasses = ...)

#unusable in spark-submit
def SaveInfo(info, file = ("model_" + str(datetime.date.today()) + "/info.txt")):
	if not os.path.exists(file):
		os.makedirs(file)
	finfo = open(file ,'a')
	finfo.write(info + "\n")
	finfo.close()
		
if __name__ == "__main__":
	sc=CreateSparkContext()
	
	StartTime = time()
	
	
	print("=============== Loading ==============")
	train_orig = LoadAndPrepare('G:\\dataset\\expedia\\train.csv')
	train_lpRDD = train_orig.map(lambda r: LabeledPoint(float(r[-1]) , r[2:13], r[15:-2]))
	#(lpRDD1, lpRDD2, lpRDD3)= train_lpRDD.randomSplit([0.7, 0.3])
	train_lpRDD.persist()
	
	test_orig = LoadAndPrepare('G:\\dataset\\expedia\\test.csv')
	test_lpRDD = test_orig.map(lambda r: LabeledPoint(float(r[-1]) , r[2:13], r[15:-2]))
	test_lpRDD.persist()	
	
	
	print("============== Training ==============")
	model = LogisticRegressionWithLBFGS.train(train_lpRDD)
	
	
	print("=============== Testing ==============")
	predictions = model.predict(test_lpRDD.map(lambda x: x.features))
	
	labelsAndPredictions = test_lpRDD.map(lambda lp: lp.label).zip(predictions)
	testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_lpRDD.count())
	Duration = time() - StartTime
	
	
	print("============= Done & Saving ==========")
	SavePath = "model_" + str(datetime.date.today()) + "-" + str(int(StartTime))
	
	train_count = train_lpRDD.count(); test_count = test_lpRDD.count();
	
	
	print(str(datetime.date.today()) + "-" + str(StartTime))
	print("DataSplit=none(WHOLE)")
	print("train="+str(train_count))
	print("test="+str(test_count))
	#print("lpRDD1="+str(lp1_count))
	#print("lpRDD2="+str(lp2_count))
	print("Duration="+str(Duration))
	print("ErrorRate=" + str(testErr * 100) + "%")
	SaveOrLoadModel(model , option = 's', path = SavePath)
	
	finfo = open('info.txt','a')
	finfo.write((str(datetime.date.today()) + "-" + str(StartTime)) + "\n")
	finfo.write("DataSplit=none(WHOLE)" + "\n")
	finfo.write("train="+str(train_count) + "\n")
	finfo.write("test="+str(test_count) + "\n")
	# finfo.write("lpRDD1="+str(lp1_count) + "\n")
	# finfo.write("lpRDD2="+str(lp2_count) + "\n")
	finfo.write("Duration="+str(Duration) + "\n")
	finfo.write("ErrorRate=" + str(testErr * 100) + "%" + "\n\n")
	finfo.close()
	
	'''usable in interaction EXEC but unusable in spark-submit'''
	SaveInfo((str(datetime.date.today()) + "-" + str(StartTime)))
	SaveInfo("DataSplit=none(WHOLE)", file = (SavePath + "\\info.txt"))
	SaveInfo("train="+str(train_count), file = (SavePath + "\\info.txt"))
	SaveInfo("test="+str(test_count), file = (SavePath + "\\info.txt"))
	# SaveInfo("lpRDD1="+str(lp1_count), file = (SavePath + "\\info.txt"))
	# SaveInfo("lpRDD2="+str(lp2_count), file = (SavePath + "\\info.txt"))
	SaveInfo("Duration="+str(Duration), file = (SavePath + "\\info.txt"))
	SaveInfo("ErrorRate=" + str(testErr * 100) + "%\n", file = (SavePath + "\\info.txt"))

	
	sc.stop()
