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
	powersum = np.power(randomx,2)+np.power(randomy,2)
	hit2=np.count_nonzero(powersum<=1)
	circlex = randomx[np.nonzero(powersum<=1)]
	circley = randomy[np.nonzero(powersum<=1)]
	outx = randomx[np.nonzero(powersum>1)]
	outy = randomy[np.nonzero(powersum>1)]
	#print(circlex)
	#circlex = np.array(1)
	#circley = np.array(1)
	#outx = np.array(1)
	#outy = np.array(1)
	#plt.axis([-1,1,-1,1])
	'''
	for i in range(0,10**count):
		if (powersum[i] <= 1):
			hit=hit+1;
			#plt.plot(randomx[i],randomy[i],'bo')
			circlex = np.append(circlex,randomx[i])
			circley = np.append(circley,randomy[i])

		else:
			#plt.plot(randomx[i],randomy[i],'ro')
			outx=np.append(outx,randomx[i])
			outy=np.append(outy,randomy[i])
	'''
	plt.plot(circlex,circley,'bo')
	plt.plot(outx,outy,'ro')

		

	
	#plt.title('number of points: pi value: %.3f' %pi)
	pi = 4.0*hit2/x
	print(pi)
plt.show()
