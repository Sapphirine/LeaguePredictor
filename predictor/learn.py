from ConfigParser import RawConfigParser

config = RawConfigParser()
config.read('config.ini')

RIOT_API_KEY = config.get('riot_games', 'API_KEY')

class Predictor:
	def __init__(self, name, trainFn, predictFn, setupFn):
		self.name = name
		self.trainFn = trainFn
		self.predictFn = predictFn
		self.setupFn = setupFn

	def setup(self, matches):
		return self.setupFn(matches) if self.setupFn else matches

	def train(self, matches):
		self.model = self.trainFn(matches)

	def predict(self, matches):
		return self.predictFn(self.model, matches)

	def evaluate(self, matches, fn = None):
		if fn:
			return matches.map(fn).zip(self.predict(matches)).map(lambda (match, prediction): prediction * (1 if winner(match) else -1))
		return matches.zip(self.predict(matches)).map(lambda (match, prediction): prediction * (1 if winner(match) else -1))

def bestNextChamp(sc, model, pickNum, selectedChamps, roles = None, champions = None):
	import numpy
	from pyspark.mllib.linalg import SparseVector

	if not champions:
		champions = championList()

	size = max(champions) + 1

	if roles:
		size *= 5

		for i in range(len(roles)):
			selectedChamps[i] = selectedChamps[i] + roles[i] * (max(champions) + 1)

	ss = {}

	for c in selectedChamps[: (pickNum + 1) / 2]:
		ss[c] = 1
	for c in selectedChamps[5 : 5 + pickNum / 2]:
		ss[c] = -1

	partialVect = SparseVector(size, ss)
	s = sum(partialVect.values)
	sign = 1 if s <= 0 else -1
	available = list(champions)

	for champ in partialVect.indices:
		available.remove(champ % (max(champions) + 1))

	def f(champ):
		i = 0
		newVects = []

		while champ + i * (max(champions) + 1) < len(partialVect):
			newVect = SparseVector(len(partialVect), partialVect.indices, partialVect.values)
			newVect.indices = numpy.append(newVect.indices, [champ + i * (max(champions) + 1)])
			newVect.values = numpy.append(newVect.values, [sign])
			newVects.append(newVect)
			i += 1

		return newVects

	champsRdd = sc.parallelize(available)
	vectors = champsRdd.flatMap(f)

	if hasattr(model, '_threshold'):
		t = model._threshold
		model.clearThreshold()
		predicted = model.predict(vectors).map(lambda x: x - t)
		predicted.collect()
		model._threshold = t
	else:
		predicted = model.predict(vectors)

	zipped = champsRdd.map(lambda x: [x]).zip(predicted)

	def g(x, y):
		if sign * x[1] < sign * y[1]:
			return y[0], y[1]
		elif sign * x[1] == sign * y[1]:
			return x[0] + y[0], x[1]
		else:
			return x

	return zipped.reduce(g)

def winner(match):
	return match.label if hasattr(match, 'label') else ((match['teams'][0]['teamId'] if match['teams'][0]['winner'] else match['teams'][1]['teamId']) == 100)

# def championList():
#     import json, urllib2
#     data = json.loads(urllib2.urlopen('https://na.api.pvp.net/api/lol/static-data/na/v1.2/champion?&api_key={0}'.format(RIOT_API_KEY)).read())['data']
#     champions = {}

#     for champ in data:
#         champions[data[champ]['id']] = data[champ]['name']

#     return champions

def championList():
	import json

	with open('champions') as input:
		data = json.loads(input.read())['data']

	champions = {}

	for champ in data:
		champions[int(data[champ]['key'])] = data[champ]['name']

	return champions

def predictWithModel(model, selectedChamps, champions = None, roles = None):
	from pyspark.mllib.linalg import SparseVector

	if not champions:
		champions = championList()

	size = max(champions) + 1

	if roles:
		size *= 5

		for i in range(len(roles)):
			selectedChamps[i] = selectedChamps[i] + roles[i] * (max(champions) + 1)

	selectedChamps.sort()

	s = SparseVector(size, selectedChamps, [1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

	if hasattr(model, '_threshold'):
		t = model._threshold
		model.clearThreshold()
		x =  model.predict(s)
		model._threshold = t
		return x - t
	
	return model.predict(s)

def main(sc, shortcut = True):
	import json
	from pyspark.mllib.classification import SVMWithSGD, LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
	from pyspark.mllib.tree import DecisionTree, RandomForest#, GradientBoostedTrees
	from pyspark.mllib.linalg import SparseVector
	from pyspark.mllib.regression import LabeledPoint
	from sys import stdout

	####################################
	def ACWR_train(matches):
		def champInfo(match):
			winner = match['teams'][0]['teamId'] if match['teams'][0]['winner'] else match['teams'][1]['teamId']
			return [(player['championId'], player['teamId'] == winner) for player in match['participants']]

		champStats = matches.flatMap(champInfo)
		champWins = champStats.reduceByKey(lambda x, y: x + y)
		champGames = champStats.countByKey()
		champWinRates = {}

		for champ in champions:
			wins = champWins.filter(lambda x: x[0] == champ).map(lambda x: x[1]).collect()
			
			if len(wins) == 0:
				wins = [0]

			assert len(wins) == 1
			champWinRates[champ] = float(wins[0]) / champGames[champ]

		return champWinRates

	def ACWR_predict(champWinRates, match):
		blueAvg, purpleAvg = 0, 0

		for player in match['participants']:
			if player['teamId'] == 100:
				blueAvg += champWinRates[player['championId']]
			elif player['teamId'] == 200:
				purpleAvg += champWinRates[player['championId']]

		winner = match['teams'][0]['teamId'] if match['teams'][0]['winner'] else match['teams'][1]['teamId']
		return (blueAvg - purpleAvg) / (blueAvg + purpleAvg)
	####################################

	####################################
	def toVector(match):
		vec = {}
		winner = match['teams'][0]['teamId'] if match['teams'][0]['winner'] else match['teams'][1]['teamId']

		for player in match['participants']:
			if player['teamId'] == 100:
				vec[player['championId']] = 1
			elif player['teamId'] == 200:
				vec[player['championId']] = -1

		return LabeledPoint(winner == 100, SparseVector(max(champions) + 1, vec))

	def toVector2(match):
		def getRole(player):
			lane = player['timeline']['lane']
			role = player['timeline']['role']

			if lane == 'JUNGLE' and role == 'NONE':
				return 0#'jng'
			if lane == 'TOP' and role == 'SOLO':
				return 1#'top'
			if lane == 'MIDDLE' and role == 'SOLO':
				return 2#'mid'
			if lane == 'BOTTOM' and role == 'DUO_CARRY':
				return 3#'adc'
			if lane == 'BOTTOM' and role == 'DUO_SUPPORT':
				return 4#'sup'

			return -1

		winner = match['teams'][0]['teamId'] if match['teams'][0]['winner'] else match['teams'][1]['teamId']
		vec = {}
		vecLen = max(champions) + 1
		roles = set()

		for player in match['participants']:
			role = getRole(player)

			if role < 0:
				return []

			vec[role * vecLen + player['championId']] = 1 if player['teamId'] == 100 else -1
			roles.add((role+1) * (1 if player['teamId'] == 100 else -1))

		if len(roles) != 10:
			return []
		else:
			return [LabeledPoint(winner == 100, SparseVector(vecLen * 5, vec))]
	####################################

	####################################
	def predict(model, vectors):
		t = model._threshold
		model.clearThreshold()
		ret = vectors.map(lambda v: model.predict(v.features) - t)
		ret.collect() # to force prediction now
		model._threshold = t
		return ret

	def tree(model, vectors):
		return model.predict(vectors.map(lambda x: x.features)).map(lambda x: 2 * x - 1)
	####################################
	
	####################################
	def stats(results, output = stdout):
		results.cache()
		size = results.count()
		marginSum = results.reduce(lambda x, y: x+y)
		maxCorrect = results.reduce(lambda x, y: max(x, y))
		maxWrong = results.reduce(lambda x, y: min(x, y))
		totalMargin = results.reduce(lambda x, y: abs(x) + abs(y))
		totalPosMargin = results.filter(lambda x: x > 0).reduce(lambda x,y: x+y)
		totalNegMargin = results.filter(lambda x: x < 0).reduce(lambda x,y: x+y)
		sizePos = results.filter(lambda x: x > 0).count()
		sizeNeg = results.filter(lambda x: x < 0).count()

		avgMargin = totalMargin / size
		avgPosMargin = totalPosMargin / sizePos
		avgNegMargin = totalNegMargin / sizeNeg

		print >> output, 'correct: {0}, total: {1}, acc: {2}'.format(sizePos, size, float(sizePos) / size)
		print >> output, 'margin sum: {0}, margin sum / size: {1}'.format(marginSum, marginSum / size)
		print >> output, 'max positive margin: {0}, max negative margin: {1}'.format(maxCorrect, maxWrong)
		print >> output, 'avg margin: {0}, avg pos margin: {1}, avg neg margin: {2}'.format(avgMargin, avgPosMargin, avgNegMargin)

		return float(sizePos) / size, avgMargin
	####################################

	acwr = Predictor('ACWR - baseline', trainFn = ACWR_train, predictFn = lambda model, matches: matches.map(lambda match: ACWR_predict(model, match)), setupFn = None)

	logReg1 = Predictor('Logistic Regression with SGD 1', trainFn = LogisticRegressionWithSGD.train, predictFn = predict, setupFn = lambda matches: matches.map(toVector))
	logReg2 = Predictor('Logistic Regression with SGD 2', trainFn = LogisticRegressionWithSGD.train, predictFn = predict, setupFn = lambda matches: matches.flatMap(toVector2))

	logReg3 = Predictor('Logistic Regression with LBGFS 1', trainFn = LogisticRegressionWithLBFGS.train, predictFn = predict, setupFn = lambda matches: matches.map(toVector))
	logReg4 = Predictor('Logistic Regression with LBGFS 2', trainFn = LogisticRegressionWithLBFGS.train, predictFn = predict, setupFn = lambda matches: matches.flatMap(toVector2))

	svm1 = Predictor('SVM 1', trainFn = SVMWithSGD.train, predictFn = predict, setupFn = lambda matches: matches.map(toVector))
	svm2 = Predictor('SVM 2', trainFn = SVMWithSGD.train, predictFn = predict, setupFn = lambda matches: matches.flatMap(toVector2))

	dtree1 = Predictor('Decision Tree 1', trainFn = lambda vectors: DecisionTree.trainClassifier(vectors, 2, {}), predictFn = tree, setupFn = lambda matches: matches.map(toVector))
	dtree2 = Predictor('Decision Tree 2', trainFn = lambda vectors: DecisionTree.trainClassifier(vectors, 2, {}), predictFn = tree, setupFn = lambda matches: matches.flatMap(toVector2))

	dtree3 = Predictor('Random Forests 1', trainFn = lambda vectors: RandomForest.trainClassifier(vectors, 2, {}, 10), predictFn = tree, setupFn = lambda matches: matches.map(toVector))
	dtree4 = Predictor('Random Forests 2', trainFn = lambda vectors: RandomForest.trainClassifier(vectors, 2, {}, 10), predictFn = tree, setupFn = lambda matches: matches.flatMap(toVector2))

	def aggTrainer(trainFns, vectors):
		return [ fn(vectors) for fn in trainFns ]
	def aggPredict(models, predictFns, vectors):
		preds = vectors.map(lambda x: 0)

		for i in range(len(models)):
			preds = preds.zip(predictFns[i](models[i], vectors)).map(lambda (x, y): x + (1 if y > 0 else -1))

		return preds

	# trains = [LogisticRegressionWithSGD.train]#, LogisticRegressionWithLBFGS.train, SVMWithSGD.train]
	# predicts = [predict]#, predict, predict]

	# combo1 = Predictor('Combo 1', trainFn = lambda vectors: aggTrainer(trains, vectors), predictFn = lambda model, vectors: aggPredict(model, predicts, vectors), setupFn = lambda matches: matches.flatMap(toVector))
	# combo2 = Predictor('Combo 2', trainFn = lambda vectors: aggTrainer(trains, vectors), predictFn = lambda model, vectors: aggPredict(model, predicts, vectors), setupFn = lambda matches: matches.flatMap(toVector2))
	
	if shortcut:
		predictors1 = [logReg3]
		predictors2 = [logReg4]
	else:
		predictors1 = [ acwr, logReg1, logReg3, svm1, dtree1, dtree3 ]
		predictors2 = [ logReg2, logReg4, svm2, dtree2, dtree4 ]

	traingingMatches = sc.textFile('data/training/*.json').map(json.loads).flatMap(lambda x: x['matches'])
	testingMatches = sc.textFile('data/test/*.json').map(json.loads).flatMap(lambda x: x['matches'])
	traingingMatches.cache()
	testingMatches.cache()
	champions = championList()

	bestPredictor1, accuracy1 = None, 0
	bestPredictor2, accuracy2 = None, 0

	with open('out2' if shortcut else 'out', 'w') as output:
		for p in predictors1:
			vectors = p.setup(traingingMatches)
			p.train(vectors)
			print >> output, p.name
			print >> output, '--------------------------'
			print >> output, 'training evaluation'
			stats(p.evaluate(vectors, None if p.setupFn else toVector), output)
			print >> output
			print >> output, 'testing evaluation'
			acc, margin = stats(p.evaluate(p.setup(testingMatches), None if p.setupFn else toVector), output)
			print >> output, '--------------------------'
			print >> output


			if acc > accuracy1:
				bestPredictor1 = p.model, margin

		for p in predictors2:
			vectors = p.setup(traingingMatches)
			p.train(vectors)
			print >> output, p.name
			print >> output, '--------------------------'
			print >> output, 'training evaluation'
			stats(p.evaluate(vectors, None if p.setupFn else toVector), output)
			print >> output
			print >> output, 'testing evaluation'
			acc, margin = stats(p.evaluate(p.setup(testingMatches), None if p.setupFn else toVector), output)
			print >> output, '--------------------------'
			print >> output

			if acc > accuracy2:
				bestPredictor2 = p.model, margin

	return bestPredictor1, bestPredictor2

if __name__ == '__main__':
	import sys, os

	sparkHome = config.get('spark', 'SPARK_HOME')

	sys.path.append('{0}/python/build'.format(sparkHome))
	sys.path.append('{0}/python'.format(sparkHome))
	os.environ['SPARK_HOME'] = sparkHome

	from pyspark import SparkContext

	main(SparkContext(), False)
