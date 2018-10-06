import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

def plot_beta_distribution(a,b,name,points=100):
    '''
    This function will plot the beta distribution give the parameters a & b
    '''
    #Getting the normalixation constant
    constant=gamma(a+b)/(gamma(a)*gamma(b))

    #Sampling the points uniformly
    delta_x=1.0/points
    x=np.arange(0,1+delta_x,delta_x)

    #Now calculating the pdf on these points
    beta=constant*(x**(a-1))*((1-x)**(b-1))

    #Plotting the pdf
    plt.plot(x,beta)
    plt.xlabel('mu (u)')
    plt.ylabel('probability_density')
    plt.title('Probability Density Function : P(u|a,b)')
    #plt.show()
    plt.savefig(name)
    plt.clf()


def iterative_learning(prior_a,prior_b,toss_sample,plot_control=10):
    #Getting the metadata
    N=toss_sample.shape[0]

    #Plotting the initial prior distribution
    plot_beta_distribution(prior_a,prior_b,name='prior.png')
    #Now going iteratively and plotting the beta distribution
    for i in range(N):
        #Calculating the new posteror based on the current evidence of the data
        posterior_a = prior_a+toss_sample[i]
        posterior_b = prior_b+(1-toss_sample[i])

        #Plotting the new distribution
        if(i%plot_control==0):
            plot_beta_distribution(posterior_a,posterior_b,name=str(i)+'.png')

        #Updating the current posterir as prior for next iteration
        prior_a=posterior_a
        prior_b=posterior_b


def one_shot_learning(prior_a,prior_b,toss_sample):
    '''
    This function will give the final posterior distribution by seeing the
    whole data at once.
    '''
    #Getting the metadata
    N=toss_sample.shape[0]
    correction_m=np.sum(toss_sample)
    correction_l=N-correction_m

    #Getting the posteror parameters of the distribution
    posterior_a = prior_a+correction_m
    posterior_b = prior_b+correction_l

    #Plotting the final distribution
    plot_beta_distribution(posterior_a,posterior_b,name='one_shot.png')


if __name__=='__main__':
    ############ CONTROL VARIABLES ##############
    dataset_size=150
    #Parameters for the prior
    a=2
    b=int((0.6/0.4)*a)

    ########### MAIN CODE ######################
    #Sampling the dastaset randomly
    toss_sample=np.random.randint(0,2,size=dataset_size)

    #Plotting the distibution iteratively
    print "Plotting the Iterative solution"
    iterative_learning(a,b,toss_sample,plot_control=1)

    #Plotting the distribution for the one-shot learning
    print "Printing the One-Shot Solution"
    one_shot_learning(a,b,toss_sample)
