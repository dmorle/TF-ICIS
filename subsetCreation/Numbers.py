import sys
import math

def product(ans, num):
	ans = ans.__copy__()
	if type(num) is int:
		ans.n*=num
		return ans
	elif type(num) is rational:
		ans.n*=num.n
		ans.d*=num.d
		ans.reduce()
		return ans
	else:
		print("error")
		sys.exit(1)
	return

class rational:
	def __init__(self, numerator, denominator):
		self.n = numerator
		self.d = denominator
	def __copy__(self):
		return type(self)(self.n, self.d)
	def reduce(self):
		negative = False
		if (self.n < 0 or self.d < 0) and not (self.n < 0 and self.d < 0):
			negative = True
		self.n = abs(self.n)
		self.d = abs(self.d)
		if self.n != 0:
			test = []
			low = min(self.n, self.d)
			lim = int(math.sqrt(low))+1
			while low%2 == 0:
				low/=2
				test.append(2)
			for i in range(3, lim, 2):
				while low%i == 0:
					low/=i
					test.append(i)
			if low > 2:
				test.append(low)
			if self.n > self.d:
				for i in test:
					if self.n%i == 0:
						self.n/=i
						self.d/=i
			else:
				for i in test:
					if self.d%i == 0:
						self.n/=i
						self.d/=i
			self.n = int(self.n)
			self.d = int(self.d)
			if negative:
				self.n *= -1
		else:
			self.d = 1
		return
	def sum(self, num):
		if type(num) is int:
			self.n += num*self.d
		elif type(num) is rational:
			self.n = self.n*num.d + num.n*self.d
			self.d *= num.d
			self.reduce()
		else:
			print("error")
			sys.exit(1)
		return()
	def difference(self, num):
		if type(num) is int:
			self.n -= num*self.d
		elif type(num) is rational:
			self.n = self.n*num.d - num.n*self.d
			self.d *= num.d
			self.reduce()
		else:
			print("error")
			sys.exit(1)
		return()
	def product(self, num):
		if type(num) is int:
			self.n*=num
		elif type(num) is rational:
			self.n*=num.n
			self.d*=num.d
		else:
			print("error")
			sys.exit(1)
		self.reduce()
		return()
	def quotient(self, num):
		if type(num) is int:
			self.d*=num
		elif type(num) is rational:
			if num.n == 0:
				raise(ZeroDivisionError())
			self.n*=num.d
			self.d*=num.n
			if self.d < 0:
				self.d*=-1
				self.n*=-1
		else:
			print("error")
			sys.exit(1)
		self.reduce()
		return()
	def pow(self, num):
		self.n = self.n**num
		self.d = self.d**num
		self.reduce()
		return
	def getVal(self):
		return self.n/self.d
	def print(self):
		print(str(self.n)+"/"+str(self.d))

class term:
	def __init__(self, coeifficient=1, power=0, base=1):
		self.coeifficient = coeifficient
		self.power = power
		self.base = base
	
	def getString(self):
		string = ""
		if self.power == 0 and self.base == 1:
			if type(coeifficient) is rational:
				string = str(self.coeifficient.n) + "/" + str(self.coeifficient.d)
			else:
				string = str(self.coeifficient)
		elif self.base == 1:
			if type(self.coeifficient) is rational:
				if self.coeifficient.n != 1 and self.coeifficient.d != 1:
					string = str(self.coeifficient.n) + "/" + str(self.coeifficient.d) + "*"
			else:
				if coeifficient != 1:
					string = str(self.coeifficient) + "*"
			string += "x^(" + str(self.power) + ")"
		elif self.power == 0:
			pass
		else:
			pass
		return(string)

def expand(expression):
	exp = ""
	for i in expression:
		if i != " " and i != "\n" and i != "\t":
			exp += i
	
	openBraket = False
	subExp = ""
	nExp = ""
	for i in exp:
		if openBraket:
			if i != ")":
				subExp += i
			else:
				openBraket = False
				nExp += expand(subExp) + ")"
				subExp = ""
		else:
			if i == "(":
				openBraket = True
			nExp += i
	exp = nExp
	nExp = ""

	bktSign = ''
	bktStr = ""
	for i in exp:
		if exp[i] == '(':
			bktStart = i
		if bktStart == -1:
			nExp += exp[i]
	return