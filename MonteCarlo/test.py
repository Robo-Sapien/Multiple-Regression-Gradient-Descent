from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

#fig = plt.figure()
#plt.figure(1)
for count in range(1,8):
	hit=0
	x=10**count
	ax=plt.axes(projection='3d')
	randomx = np.random.uniform(-1,1,size=x)
	randomy = np.random.uniform(-1,1,size=x)
	randomz = np.random.uniform(-1,1,size=x)
	#plt.axis([-1,1,-1,1])
	powersum = np.power(randomx,2)+np.power(randomy,2)+np.power(randomz,2)
	hit=np.count_nonzero(powersum<=1)
	spherex = randomx[np.nonzero(powersum<=1)]
	spherey = randomy[np.nonzero(powersum<=1)]
	spherez = randomz[np.nonzero(powersum<=1)]
	ax.scatter3D(spherex,spherey,spherez,c='b',marker='o')
	
	outx = randomx[np.nonzero(powersum>1)]
	outy = randomy[np.nonzero(powersum>1)]
	outz = randomz[np.nonzero(powersum>1)]
	ax.scatter3D(outx,outy,outz,c='r',marker='o')
	plt.show()
	pi = 6.0*hit/x
	#plt.title('number of points: pi value: %.3f' %pi)
	#plt.shhow()
	print(pi)
#plt.show()
