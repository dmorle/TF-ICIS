import copy
import Numbers

class hilbertCurve:
	def __init__(self, depth, base = [[0, 1], [1, 0], [0, -1]]):
		self.curve = base
		self.size = int((len(base)+1)**(1/2)) - 1
		for i in range(depth):
			self.size = self.size*2 + 1
		for i in range(depth):
			self.incrimentDepth()
		self.calculatePoints()
		return

	def incrimentDepth(self):
		L = len(self.curve)
		c00 = copy.deepcopy(self.curve)
		for move in c00:
			(move[0], move[1]) = (move[1], move[0])
		c01 = copy.deepcopy(self.curve)
		c10 = [None] * L
		i = L
		for move in c00:
			i -= 1
			c10[i] = [move[0], -move[1]]
		c11 = [None] * L
		i = L
		for move in c01:
			i -= 1
			c11[i] = [move[0], -move[1]]
		self.curve = c00 + [[0, 1]] + c01 + [[1, 0]] + c11 + [[0, -1]] + c10
		del c00, c01, c10, c11
		return

	def calculatePoints(self):
		loc = [Numbers.rational(0, 1), Numbers.rational(0, 1)]
		self.rPoints = [[0, 0]]
		self.dPoints = [[0.0, 0.0]]
		for move in self.curve:
			loc[0].sum(Numbers.rational(move[0], self.size))
			loc[1].sum(Numbers.rational(move[1], self.size))
			self.rPoints.append(copy.deepcopy(loc))
			self.dPoints.append([loc[0].getVal(), loc[1].getVal()])
		return

	def getCurve(self):
		return self.curve

	def getRPoints(self):
		return self.rPoints

	def getDPoints(self):
		return self.dPoints
