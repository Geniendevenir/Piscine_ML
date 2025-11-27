class TinyStatistician:
	def __init__(self):
		pass

	def mean(self, x=[]):
		if x is None or not x:
			return None

		result = 0
		for elem in x:
			result += elem
		result /= len(x)
		return result

	def median(self, x):
		if x is None or not x:
			return None
		x.sort()
		if len(x) % 2 == 0:
			i = int((len(x) + 1) / 2)
			return float(x[i])
		else:
			i = int(len(x) / 2)
			return float(x[i])

	def quartile(self, x):
		if x is None or not x:
			return None
		x.sort()
		if len(x) % 2 == 0:
			firstQ = self.median(x[:int(len(x) / 2)])
			thirdQ = self.median(x[int(len(x) / 2):])
		else:
			firstQ = self.median(x[:int((len(x) - 1) / 2)])
			thirdQ = self.median(x[int((len(x) - 1) / 2):])
		return [firstQ, thirdQ]

	def var(self, x):
		if x is None or not x:
			return None

		m = self.mean(x)
		result = []

		for elem in x:
			result.append(pow(elem - m, 2))
		return self.mean(result)

	def std(self, x):
		if x is None or not x:
			return None

		return pow(self.var(x), 0.5)

tstat = TinyStatistician()
a = [1, 42, 300, 10, 59]

print(tstat.mean(a))
# Expected result: 82.4

print(tstat.median(a))
# Expected result: 42.0

print(tstat.quartile(a))
# Expected result: [10.0, 59.0]

print(tstat.var(a))
# Expected result: 12279.439999999999

print(tstat.std(a))
# Expected result: 110.81263465868862