import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

#fig = plt.figure()
#plt.figure(1)
plt.figure(1,figsize=(13,12))
print('\n')
print('Printing pi values for Part A: Circle (2D)')
for count in range(1,8):
	hit=0
	N=10**count
	plt.subplot(3,3,count)
	randomx = np.random.uniform(-1,1,size=N)
	randomy = np.random.uniform(-1,1,size=N)
	plt.axis([-1,1,-1,1])
	powersum = np.power(randomx,2)+np.power(randomy,2)
	hit=np.count_nonzero(powersum<=1)
	pi = 4.0*hit/N

	circlex = randomx[np.nonzero(powersum<=1)]
	circley = randomy[np.nonzero(powersum<=1)]

	outx = randomx[np.nonzero(powersum>1)]
	outy = randomy[np.nonzero(powersum>1)]
	#plt.plot(circlex,circley,'bo')
	#plt.plot(outx,outy,'ro')	
	
	plt.title(r'N: $10^%i$ pi : %f' %(count,pi))

	print('for n = %i, approx pi = %f' %(N,pi))
#plt.show()

print('\n')
print('Printing pi values for Part B: Sphere (3D)')
plt.figure(2)
for count in range(1,8):
	hit=0
	N=10**count
	ax=plt.axes(projection='3d')
	randomx = np.random.uniform(-1,1,size=N)
	randomy = np.random.uniform(-1,1,size=N)
	randomz = np.random.uniform(-1,1,size=N)
	#plt.axis([-1,1,-1,1])
	powersum = np.power(randomx,2)+np.power(randomy,2)+np.power(randomz,2)
	hit=np.count_nonzero(powersum<=1)
	pi = 6.0*hit/N

	spherex = randomx[np.nonzero(powersum<=1)]
	spherey = randomy[np.nonzero(powersum<=1)]
	spherez = randomz[np.nonzero(powersum<=1)]
	
	outx = randomx[np.nonzero(powersum>1)]
	outy = randomy[np.nonzero(powersum>1)]
	outz = randomz[np.nonzero(powersum>1)]
	#ax.scatter3D(spherex,spherey,spherez,c='b',marker='o')
	#ax.scatter3D(outx,outy,outz,c='r',marker='o')
	
	plt.title(r'N: $10^%i$ pi : %f' %(count,pi))
	print('for n = %i, approx pi = %f' %(N,pi))
	#plt.show()
#plt.show()