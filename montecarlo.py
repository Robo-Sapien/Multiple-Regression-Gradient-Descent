import matplotlib.pyplot as plt
import numpy as np

for count in range(1,8):
	hit=0
	x=10**count
	randomx = np.random.uniform(-1,1,size=x)
	randomy = np.random.uniform(-1,1,size=x)
	#plt.axis([-1,1,-1,1])

	for i in range(0,10**count):
		if (randomx[i]**2 + randomy[i]**2 <= 1):
			hit=hit+1;
			#plt.plot(randomx[i],randomy[i],'bo')
		else:
			#plt.plot(randomx[i],randomy[i],'ro')
			y=x

	pi = 4.0*hit/x
	#plt.title('number of points: pi value: %.3f' %pi)
	#plt.show()
	print(pi)


for count in range(1,8):
	hit=0
	x=10**count
	randomx = np.random.uniform(-1,1,size=x)
	randomy = np.random.uniform(-1,1,size=x)
	randomz = np.random.uniform(-1,1,size=x)
	#plt.axis([-1,1,-1,1])

	for i in range(0,10**count):
		if (randomx[i]**2 + randomy[i]**2 + randomz[i]**2 <= 1):
			hit=hit+1;
			#plt.plot(randomx[i],randomy[i],'bo')
		else:
			#plt.plot(randomx[i],randomy[i],'ro')
			y=x
	pi = 6.0*hit/x
	#plt.title('number of points: pi value: %.3f' %pi)
	#plt.show()
	print(pi)