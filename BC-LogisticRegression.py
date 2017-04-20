#REF:https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/binary_classification_metrics_example.py
from pyspark import SparkConf, SparkContext
from time import time, strftime, localtime
import os
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel, NaiveBayes, NaiveBayesModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics	#wait fot TESTing , not in use
 
def CreateSparkContext():
    sparkConf = SparkConf()                                            \
                         .setAppName("NCCUBigDataProj1")           \
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

def gettime(t = time(), method = ""):
	if(method == "f"):	#f means floder
		return strftime("%Y-%m-%d-%H-%M-%S(+0800)", localtime(t))
	else:
		return strftime("%Y-%m-%d %H:%M:%S(+0800)", localtime(t))

def LoadAndPrepare(dataset):
	f = sc.textFile(dataset).map(lambda x: str(x).replace('NULL','0'))
	header = f.first()
	f = f.filter(lambda x: x != header).map(lambda x:x.split(","))\
		.map(lambda r: LabeledPoint(float(r[-1]) , (r[2:13] + r[15:-4])))
	#for training data, [0], [1], [14], [51], [52] not in features, [53]=[-1] is label
	return f

def ModelPredict(model, dataset):
	return model.predict(dataset.map(lambda x: x.features))

def SaveOrLoadModel(model = "" , folder = ("model_" + str(time())) , option = 's'):
	if (option == 's'):
		model.save(sc, folder)
	elif (option == 'l'):
		return LogisticRegressionModel.load(sc, folder)

#unuse
def SaveInfo(info, file = ("info.txt")):
	# if not os.path.exists(file):
		# os.makedirs(file)
	finfo = open(file ,'a')
	finfo.write(info + "\n")
	finfo.close()

if __name__ == "__main__":
	sc = CreateSparkContext()
	StartTime = time()
	SavePath = gettime(StartTime, "f")
	
	train_lpRDDs = [0] * 4
	models = []
	predictions = []
	PredictionsAndLabels = []
	ErrCount = []
	ModelMetrics = []
	trainCounts = []
	
	print("============= Loading ================" + gettime(time()))
	(train_lpRDDs[0], train_lpRDDs[1], train_lpRDDs[2], train_lpRDDs[3], Validtion_lpRDD) \
		= LoadAndPrepare('G:\\dataset\\expedia\\train.csv').randomSplit([1, 1, 1, 1, 1])
	'''
		#These codes NEED MORE DEBUG to compute the correct results!
		#for training data, [0], [1], [14], [51], [52] not in features, [53]=[-1] is label
		train_lpRDD = LoadAndPrepare('G:\\dataset\\expedia\\train.csv')\
			.map(lambda r: LabeledPoint(float(r[-1]) , (r[2:13] + r[15:-4])))
		train_lpRDD.persist()
			#for testing data, thereis no [14], [51] , [52] ,[53]not in features, and there is NO label
		test_lpRDD = LoadAndPrepare('G:\\dataset\\expedia\\test.csv')\
			.map(lambda r: LabeledPoint(float(r[-1]) , r[2:-4]))
		test_lpRDD.persist()
	'''
	print("============= Training ===============" + gettime(time()))
	models = [LogisticRegressionWithLBFGS.train(train_lpRDDs[0], iterations = 200) , \
			  DecisionTree.trainClassifier(train_lpRDDs[1], numClasses=2, categoricalFeaturesInfo={},\
						impurity='gini', maxDepth=30, maxBins=32) , \
			  LinearRegressionWithSGD.train(train_lpRDDs[2]) , \
			  LinearRegressionWithSGD.train(train_lpRDDs[3])]
			  # NaiveBayes.train(train_lpRDDs[2], 1.0) , \
	for i in range(len(models)):
		SaveOrLoadModel(models[i], folder = "model_" + SavePath + "\\" + str(i+1), option = 's')
	'''
	#Decision Tree
	model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},\
						impurity='gini', maxDepth=5, maxBins=32)
	NaiveBayes
	model = NaiveBayes.train(traindata, 1.0)
	Linear regression
	model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001)
	'''
	
	print("============= Predicting =============" + gettime(time()))
	for i in range(len(models)):
		predictions.append(ModelPredict(models[i], Validtion_lpRDD))

	print("============= Computing ==============" + gettime(time()))
	for i in range(len(models)):
		PredictionsAndLabels.append(predictions[i].map(lambda x:float(x)).zip(Validtion_lpRDD.map(lambda lp: lp.label)))
		ErrCount.append(PredictionsAndLabels[i].filter(lambda lp: (lp[0] != lp[1])).count())
		ModelMetrics.append(BinaryClassificationMetrics(PredictionsAndLabels[i]))
		trainCounts.append(train_lpRDDs[i].count())
	ValidtionCount = Validtion_lpRDD.count()
	
	ModelingDuration = time() - StartTime
	
	print("============= Saving =================" + gettime(time()))
	finfo = open('info.txt' , 'a')
	finfo.write(str(gettime(StartTime)) + "\n")
	finfo.write("DataSplit        :none(WHOLE)" + "\n")
	for i in range(len(models)):
		finfo.write("train" + str(i+1) + "           :" + str(trainCounts[i]) + "\n")
		finfo.write("\ttrain" + str(i+1) + "Err    :" + str(ErrCount[i]) + "\n")
		finfo.write("\ttrain" + str(i+1) + "AUC    :" + str(ModelMetrics[i].areaUnderROC) + "\n")#
		finfo.write("\ttrain" + str(i+1) + "APR    :" + str(ModelMetrics[i].areaUnderPR) + "\n")#
	finfo.write("Validation       :" + str(ValidtionCount) + "\n")
	finfo.write("MinimumErrorCount:" + str(min(ErrCount)) + "\n")
	finfo.write("MinimumErrorRate :" + str(float(min(ErrCount)) / ValidtionCount * 100) + "%\n")
	finfo.write("ModelingDuration :"+str(ModelingDuration) + "\n")
	finfo.write("TotalDuration :"+str(time()-StartTime) + "\n\n")
	finfo.close()
	'''fail while give parameter2 == (AnyPath + "\\info.txt"))'''
	
	for i in range(len(PredictionsAndLabels)):
		fPrediction = open(("PredictionsAndLabels" + str(i + 1) + ".csv") , 'a')
		for (prediction, label) in PredictionsAndLabels[i].collect():
			fPrediction.write(str(prediction) + "," + str(label) + "\n")
		fPrediction.close()
	
	print("============= Printing ===============" + gettime(time()))
	print("DataSplit        :none(WHOLE)")
	print("Validation       :" + str(ValidtionCount))
	for i in range(len(models)):
		print("train" + str(i+1) + "           :" + str(trainCounts[i]))
		print("train" + str(i+1) + "Err        :" + str(ErrCount[i]))
		print("train" + str(i+1) + "AUC        :" + str(ModelMetrics[i].areaUnderROC))
		print("train" + str(i+1) + "APR        :" + str(ModelMetrics[i].areaUnderPR))
	print("MinimumErrorCount:" + str(min(ErrCount)))
	print("MinimumErrorRate :" + str(min(ErrCount) / ValidtionCount * 100) + "%")
	print("ModelingDuration :"+str(ModelingDuration))
	
	print("============= Done ===================" + gettime(time()))
	print("TotalDuration :"+str(time()-StartTime))
	sc.stop()
    
