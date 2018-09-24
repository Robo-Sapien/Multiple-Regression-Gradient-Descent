import matplotlib.pyplot as plt
import numpy as np

#fig = plt.figure()
plt.figure(1)
for count in range(1,8):
	hit=0
	x=10**count
	plt.subplot(2,4,count)
	randomx = np.random.uniform(-1,1,size=x)
	randomy = np.random.uniform(-1,1,size=x)
	plt.axis([-1,1,-1,1])
	powersum = np.power(randomx,2)+np.power(randomy,2)
	hit=np.count_nonzero(powersum<=1)
	circlex = randomx[np.nonzero(powersum<=1)]
	circley = randomy[np.nonzero(powersum<=1)]
	outx = randomx[np.nonzero(powersum>1)]
	outy = randomy[np.nonzero(powersum>1)]
	plt.plot(circlex,circley,'bo')
	plt.plot(outx,outy,'ro')	
	pi = 4.0*hit/x
	#plt.title('number of points: pi value: %.3f' %pi)
	print(pi)
#plt.show()


for count in range(1,8):
	hit=0
	x=10**count
	randomx = np.random.uniform(-1,1,size=x)
	randomy = np.random.uniform(-1,1,size=x)
	randomz = np.random.uniform(-1,1,size=x)
	#plt.axis([-1,1,-1,1])
	powersum = np.power(randomx,2)+np.power(randomy,2)+np.power(randomz,2)
	hit=np.count_nonzero(powersum<=1)
	spherex = randomx[np.nonzero(powersum<=1)]
	spherey = randomy[np.nonzero(powersum<=1)]
	spherez = randomz[np.nonzero(powersum<=1)]

	outx = randomx[np.nonzero(powersum>1)]
	outy = randomy[np.nonzero(powersum>1)]
	outz = randomz[np.nonzero(powersum>1)]
	
	pi = 6.0*hit/x
	#plt.title('number of points: pi value: %.3f' %pi)
	#plt.show()
	print(pi)
plt.show()