#REF:https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/binary_classification_metrics_example.py
from pyspark import SparkConf, SparkContext
import numpy as np
from time import time, strftime, localtime
import os
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionModel	#wait fot TESTing , not in use
from pyspark.mllib.evaluation import MulticlassMetrics				#wait fot TESTing , not in use
 
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
        Path="hdfs://master:9000/data/"'''

def gettime(t = time(), method = ""):
	if(method == "f"):
	#f means floder
		return strftime("%Y-%m-%d-%H-%M-%S(+0800)", localtime(t))
	else:
		return strftime("%Y-%m-%d %H:%M:%S(+0800)", localtime(t))
		
def LoadAndPrepare(dataset):
	f = sc.textFile(dataset).map(lambda x: str(x).replace('NULL','0'))
	header = f.first()
	f = f.filter(lambda x: x != header).map(lambda x:x.split(","))
	f = f.map(lambda r: LabeledPoint(float(r[-1]) , (r[2:13] + r[15:-4]+ r[-3])))
	#col[0], col[1], col[14], col[51] not in features
	return f
	
def SaveOrLoadModel(model = "" , folder = ("model_" + str(time())) , option = 's'):
	if (option == 's'):
		model.save(sc, folder)
	elif (option == 'l'):
		return LogisticRegressionModel.load(sc, folder)
	#model = LogisticRegressionModel(weights=[...], intercept=..., numFeatures = ... , numClasses = ...)

def SaveInfo(info, file = ("info.txt")):
	# if not os.path.exists(file):
		# os.makedirs(file)
	finfo = open(file ,'a')
	finfo.write(info + "\n")
	finfo.close()
		
if __name__ == "__main__":
	sc = CreateSparkContext()
	StartTime = time()
	
	print("============= Loading ================")
	train_lpRDD = LoadAndPrepare('G:\\dataset\\expedia\\train.csv')
	train_lpRDD.persist()
	test_lpRDD = LoadAndPrepare('G:\\dataset\\expedia\\test.csv')
	test_lpRDD.persist()	
	
	print("============= Training ===============")
	model = LogisticRegressionWithLBFGS.train(train_lpRDD)
	
	print("============= Testing ================")
	predictions = model.predict(test_lpRDD.map(lambda x: x.features))
	
	print("============= Computing ==============")
	labelsAndPredictions = test_lpRDD.map(lambda lp: lp.label).zip(predictions)
	ErrCount = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count()
	ErrRate = ErrCount / float(test_lpRDD.count())
	Duration = time() - StartTime
	# train_count = train_lpRDD.count(); test_count = test_lpRDD.count();
	
	print("============= Saving =================")
	SavePath = "model_" + gettime(StartTime, "f")
	
	SaveOrLoadModel(model , option = 's', path = SavePath)
	SaveInfo(str(gettime(StartTime)))
	SaveInfo("DataSplit=none(WHOLE)")
	SaveInfo("train=9917530")	#("train="+str(train_count))
	SaveInfo("test=6622629")	#("test="+str(test_count))
	SaveInfo("ErrorCount=" + str(ErrCount))
	SaveInfo("ErrorRate=" + str(ErrRate * 100) + "%")
	SaveInfo("Duration="+str(Duration) + "\n")
	'''fail while give parameter2 == (SavePath + "\\info.txt"))'''
	
	print("============= Printing ===============")
	print(str(gettime(StartTime)))
	print("DataSplit=none(WHOLE)")
	print("train=9917530")	#("train="+str(train_count))
	print("test=6622629")	#("test="+str(test_count))
	print("ErrorCount=" + str(ErrCount))
	print("ErrorRate=" + str(ErrRate * 100) + "%")
	print("Duration="+str(Duration))
	
	print("============= Done ===================")
	sc.stop()
