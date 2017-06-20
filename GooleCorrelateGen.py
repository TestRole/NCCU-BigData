###### -*- coding: ASCII -*-
#REF:https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/binary_classification_metrics_example.py
from __future__ import print_function
from pyspark import SparkConf, SparkContext
from time import time, strftime, localtime
import datetime
import os, sys

# Globol Parameters Here
if True:
	# FILE = '/home/user/Desktop/train_100m.csv'
	FILE = 'G:\\dataset\\kaggle_expedia-personalized-sort\\train.csv'	#for my PC @ home
	# FILE = 'D:\\dataset\\expedia\\train_15m.csv'	#for my poor Laptop

def CreateSparkContext():
	sparkConf = SparkConf()                                            \
						 .setAppName("NCCUBigDataProjFinal-pipe")           \
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

def Dater(DateString, Symbol = '-'):
	# The type is 'YYYY-MM-DD'
	Y = int(DateString.split(Symbol)[0])
	M = int(DateString.split(Symbol)[1])
	D = int(DateString.split(Symbol)[2])
	return datetime.date(Y, M, D)
	
def DateMagician(TheDate):
	Day1 = datetime.date(2012, 11, 4)
	if (TheDate - Day1).days <=0:
		return -1
	else:
		DateQ = (TheDate - Day1).days / 7
		# DateR = (TheDate - Day1).days % 7
		DayDistance = datetime.timedelta(days = (7 * DateQ) )
		TheDate = Day1 + DayDistance
		Y = str(TheDate.year)
		M = str(TheDate.month) if TheDate.month >= 10 else '0' + str(TheDate.month)
		D = str(TheDate.day) if TheDate.day >= 10 else '0' + str(TheDate.day)
		return Y + '-' + M + '-' + D

def DictMedianByKey(lpRDD, index_K, index_V):
	Dict = {}
	# x, y should be trans 2 strings
	# the Keys will be string @ final
	# the Value will be float @ final
	lpRDD = lpRDD.map(lambda x:(x[index_K] , x[index_V]))
	lpRDD = lpRDD.reduceByKey(lambda x,y :(str(x) + ',' + str(y)))
	lpRDD = lpRDD.map(lambda x:(x[0], sorted(list(float(a) for a in x[1].split(',')))))
	lpRDD = lpRDD.map(lambda x :(x[0], x[1][len(x[1])/2]))
	for (k, v) in lpRDD.collect():
		Dict[k] = v
	return Dict
	

if __name__ == "__main__":
	sc = CreateSparkContext()
	StartTime = time()
	SavePath = gettime(StartTime, "f")

	print("============= Processing =============" + gettime(time()))
	f = sc.textFile(FILE).map(lambda x: str(x).replace('NULL','-2')) #GIVE -2 to distinguish null
	header = f.first()
	f = f.filter(lambda x: x != header).map(lambda x : x.split(","))
	# print (f.first())
	
	'''
	# The mdeian price of the LOCATE country
	DictLocalCtyPriceGap = DictMedianByKey(f, 3, 5)

	# The mdeian price of the DESTINATION Site
	DictDestPriceGap = DictMedianByKey(f, 17, 15)
	'''

	'''
	# ===1=== The Price correlations between the proce searched and the whole country median
	# The mdeian price of the DESTINATION country
	DictDestCtyPriceGap = DictMedianByKey(f, 6, 15)
	# f = f.map(lambda x: (Dater(x[1].split()[0]) ,\
						# ((float(x[15]) - DictDestCtyPriceGap[x[6]] / (float(x[15])) if float(x[15]) !=0 else 'FAIL')),\
			  # ))
	
	# f = f.map(lambda x:    ( DateMagician(x[0]) ,x[1] ) ).filter(lambda x: (x[0] != -1 and x[1] != 'FAIL')).reduceByKey(lambda x,y: (float(x) + float(y))/2)
	'''

	# ===2=== The Price correlations between the proce searched and the whole country median
	# The mdeian price of the hotel ever
	DictEverPriceGap = DictMedianByKey(f, 7, 15)
	f = f.map(lambda x: (Dater(x[1].split()[0]) ,\
						((float(x[15]) - DictEverPriceGap[x[7]] / (float(x[15])) if float(x[15]) !=0 else 'FAIL')),\
			  ))
	
	f = f.map(lambda x:    ( DateMagician(x[0]) ,x[1] ) ).filter(lambda x: (x[0] != -1 and x[1] != 'FAIL')).reduceByKey(lambda x,y: (float(x) + float(y))/2)
	
	
	# ===3=== weekly bookings
	# f = f.map(lambda x: (DateMagician(Dater(x[1].split()[0])) , x[-1] )).filter(lambda x: (x[0] != -1)).reduceByKey(lambda x,y: (int(x) + int(y)))
	
	# ===4=== averge [14]
	# f = f.map(lambda x: (DateMagician(Dater(x[1].split()[0])) , x[14] )).reduceByKey(lambda x,y: (int(x) + int(y))/2.0)
	
	# ===5=== booking rate... fail cause the number is too small for float, so replace with counts and divide them.
	# f = f.map(lambda x: (DateMagician(Dater(x[1].split()[0])) , 1 )).reduceByKey(lambda x,y: (int(x) + int(y)))
	
	print (f.first())
	print("============= Saving =================" + gettime(time()))
	with open('GC_type.csv' , 'a') as GC:
		for each in f.collect():
			print (str(each[0]) + ',' + str(each[1])  , file = GC)

	print("============= Done ===================" + gettime(time()))
	print("TotalDuration :"+str(time()-StartTime))
	sc.stop()
	
