#REF:https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/binary_classification_metrics_example.py
from pyspark import SparkConf, SparkContext
from time import time, strftime, localtime
import os
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
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
	
	print("============= Loading ================" + gettime(time()))
	(train_lpRDD_1, train_lpRDD_2, train_lpRDD_3, train_lpRDD_4, Validtion_lpRDD)\
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
	models = [LogisticRegressionWithLBFGS.train(train_lpRDD_1) , \
				LogisticRegressionWithLBFGS.train(train_lpRDD_2) , \
				LogisticRegressionWithLBFGS.train(train_lpRDD_3) , \
				LogisticRegressionWithLBFGS.train(train_lpRDD_4)]
	for i in range(4):
		SaveOrLoadModel(models[i], folder = "model_" + SavePath + "\\" + str(i+1), option = 's')
	'''Others ML algorithms training code(temporarily saved)
		#Decision Tree
		model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},\
							impurity='gini', maxDepth=5, maxBins=32)
		NaiveBayes
		model = NaiveBayes.train(traindata)
		Linear regression
		model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001)
	'''
	
	print("============= Predicting =============" + gettime(time()))
	predictions = [\
		ModelPredict(models[0], Validtion_lpRDD) , \
		ModelPredict(models[1], Validtion_lpRDD) , \
		ModelPredict(models[2], Validtion_lpRDD) , \
		ModelPredict(models[3], Validtion_lpRDD)]
	
	print("============= Computing ==============" + gettime(time()))
	LabelsAndPredictions = [\
		Validtion_lpRDD.map(lambda lp: lp.label).zip(predictions[0]) , \
		Validtion_lpRDD.map(lambda lp: lp.label).zip(predictions[1]) , \
		Validtion_lpRDD.map(lambda lp: lp.label).zip(predictions[2]) , \
		Validtion_lpRDD.map(lambda lp: lp.label).zip(predictions[3])]
	ErrCount = [\
		LabelsAndPredictions[0].filter(lambda lp: (lp[0] != lp[1])).count() , \
		LabelsAndPredictions[1].filter(lambda lp: (lp[0] != lp[1])).count() , \
		LabelsAndPredictions[2].filter(lambda lp: (lp[0] != lp[1])).count() , \
		LabelsAndPredictions[3].filter(lambda lp: (lp[0] != lp[1])).count()]

	# ErrRate = ErrCount / float(Validtion_lpRDD.count())
	ModelingDuration = time() - StartTime
	
	train_count = [\
		train_lpRDD_1.count() , \
		train_lpRDD_2.count() , \
		train_lpRDD_3.count() , \
		train_lpRDD_4.count()]
	Validtion_count = Validtion_lpRDD.count
	
	print("============= Saving =================" + gettime(time()))
	finfo = open('info.txt' , 'a')
	finfo.write(str(gettime(StartTime)) + "\n")
	finfo.write("DataSplit=none(WHOLE)" + "\n")
	finfo.write("train_1=" + str(train_count[0]) + "\n")
	finfo.write("train_2=" + str(train_count[1]) + "\n")
	finfo.write("train_3=" + str(train_count[2]) + "\n")
	finfo.write("train_4=" + str(train_count[3]) + "\n")
	finfo.write("Validation=" + str(Validtion_count) + "\n")
	finfo.write("MinimumErrorCount=" + str(min(ErrCount)) + "\n")
	finfo.write("MinimumErrorRate=" + str(min(ErrCount) / Validtion_count * 100) + "%\n")
	finfo.write("ModelingDuration="+str(ModelingDuration) + "\n\n")
	finfo.close()
	'''fail while give parameter2 == (AnyPath + "\\info.txt"))'''
	
	for i in range(4):
		fPrediction = open(("Prediction_" + str(i) + ".csv") , 'a')
		for each in LabelsAndPredictions:
			for (label, prediction) in each.collect():
				fPrediction.write(str(label) + "," + str(prediction) + "\n")
		fPrediction.close()
	
	print("============= Printing ===============" + gettime(time()))
	print("DataSplit=none(WHOLE)")
	print("train_1=" + str(train_count[0]))
	print("train_2=" + str(train_count[1]))
	print("train_3=" + str(train_count[2]))
	print("train_4=" + str(train_count[3]))
	print("Validation=" + str(Validtion_count))
	print("MinimumErrorCount=" + str(min(ErrCount)))
	print("MinimumErrorRate=" + str(min(ErrCount) / Validtion_count * 100) + "%")
	print("ModelingDuration="+str(ModelingDuration))
	
	print("============= Done ===================" + gettime(time()))
	sc.stop()
