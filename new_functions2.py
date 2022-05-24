import autograd.numpy as np
import torch
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import TrustRegions,ConjugateGradient,ParticleSwarm,SteepestDescent, particle_swarm
from pymanopt.manifolds import Grassmann
from pymanopt import Problem
from pymanopt.tools.multi import multiprod, multitransp
import matplotlib.pyplot as plt
import tqdm
from torch.distributions import normal,laplace

def dist(X, Y):
    #distance between two Grassmann manifolds


    u, s, v = torch.linalg.svd(torch.nan_to_num(X.T@Y))
    #s[s > 1] = 1
    s = torch.arccos(s)
    #print(torch.linalg.norm(s))
    return torch.nan_to_num(torch.linalg.norm(s))

def wild_bootstrap(t,n,l):
    l = torch.tensor(l)
    W = torch.zeros(t,n)
    W[0,:] = torch.normal(0,1,(n,))
    eps = torch.normal(0,1,(t,))
    for i in range(1,t):
        W[i,:] = torch.exp(-1/l) * W[i-1,:] + torch.sqrt(1-torch.exp(-2/l)) * eps[i]
    return W


def wild_mmd(m,numBootstrap,statMatrix,l=3):

    processes = wild_bootstrap(m,numBootstrap,l)
    testStats = torch.zeros(numBootstrap)
    for process in range(numBootstrap):
        mn = torch.mean(processes[:,process])

        matFix = torch.outer(processes[:,process]-mn,processes[:,process]-mn)
        #print(matFix)


        testStats[process] =  m*(statMatrix*matFix).sum()
    
    quantile = torch.quantile(testStats,0.95)
    return quantile

def kernel_mmd(x, y, lengthscale, n_perm=1000):
    n, d = x.shape
    m, d2 = y.shape
    xy = torch.cat([x, y], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    #lengthscale = dists.median()/2
    k = torch.exp((-1 / (2*lengthscale**2)) * dists)
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    mmd = k_x.sum() / (n * (n )) + k_y.sum() / (m * (m)) - 2 * k_xy.sum() / (n * m)

    if n_perm is None:
        return mmd

    statMatrix = k_x / (n * (n )) + k_y / (m * (m)) - 2 * k_xy / (n * m)
    quantile = wild_mmd(m,n_perm,statMatrix,l=20)

    return mmd, quantile

def reduced_mmd(x, y, sigma2, alpha,num_components,n_perm=1000):

    A0 = find_p0(x,y,sigma2,num_components)
    A1 = find_p1(x,y,A0,sigma2,alpha,num_components)
    reduced_x = torch.matmul(x,A0.float())
    reduced_y = torch.matmul(y,A1.float())
    n, d = reduced_x.shape
    m, d2 = reduced_y.shape
    xy = torch.cat([reduced_x, reduced_y], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)

    k = torch.exp((-1 / (2*sigma2**2)) * dists)
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]

    mmd = k_x.sum() / (n * (n )) + k_y.sum() / (m * (m )) - 2 * k_xy.sum() / (n * m)
    if n_perm is None:
        return mmd
    statMatrix = k_x / (n * (n )) + k_y / (m * (m)) - 2 * k_xy / (n * m)
    quantile = wild_mmd(m,n_perm,statMatrix,l=20)

    return mmd, quantile

def create_cost1(x,y,sigma):
    @pymanopt.function.PyTorch
    def cost(A):
        '''
        ouput argmaxMMD^2(x,y)
        '''
        reduced_x = torch.matmul(x,A.float())
        reduced_y = torch.matmul(y,A.float()) 

        n, d = reduced_x.shape
        m, d2 = reduced_y.shape
        assert d == d2
        xy = torch.cat([reduced_x, reduced_y], dim=0)
        dists = torch.cdist(xy, xy, p=2.0)
        k = torch.exp((-1/(2*sigma**2)) * dists) 
        k_x = k[:n, :n]
        k_y = k[n:, n:]
        k_xy = k[:n, n:]
        mmd = k_x.sum() / (n * (n)) + k_y.sum() / (m * (m)) - 2 * k_xy.sum() / (n * m)

        return - mmd
    return cost

def create_cost2(x,y,A0,sigma,alpha):
    @pymanopt.function.PyTorch
    def cost(A):
        '''
        regularisation
        ouput argmaxMMD^2(x,y) - alpha*D(x,y)
        A: A_{t+w}
        A0: A_t
        '''
        reduced_x = torch.matmul(x,A.float())
        reduced_y = torch.matmul(y,A.float())
        n, d = reduced_x.shape
        m, d2 = reduced_y.shape
        assert d == d2
        xy = torch.cat([reduced_x, reduced_y], dim=0)
        dists = torch.cdist(xy, xy, p=2.0)
        k = torch.exp((-1/(2*sigma**2)) * dists) 
        k_x = k[:n, :n]
        k_y = k[n:, n:]
        k_xy = k[:n, n:]
        mmd = k_x.sum() / (n * (n)) + k_y.sum() / (m * (m)) - 2 * k_xy.sum() / (n * m)
        #print(dist(A0,A))

        regularised_mmd = - mmd + alpha * dist(A0,A) #regularisation

        return regularised_mmd
    return cost

def find_p0(x,y,sigma,num_components):
    '''
    find projector A_t
    num_components: dimension of the Grassmann manifold
    '''
    num_samples,dimension = x.shape
    manifold = Grassmann(dimension, num_components)
    mycost1 = create_cost1(x,y,sigma)
    problem = pymanopt.Problem(manifold=manifold, cost=mycost1,verbosity=0)
    #solver = TrustRegions()
    solver = ConjugateGradient()
    #solver = ParticleSwarm()
    #solver = SteepestDescent()
    A_t = solver.solve(problem)
    return torch.tensor(A_t)    

def find_p1(x,y,A0,sigma,alpha,num_components):
    #find the projector A_t+w
    num_samples,dimension = x.shape
    manifold = Grassmann(dimension, num_components)
    mycost2 = create_cost2(x,y,A0,sigma,alpha)
    problem = pymanopt.Problem(manifold=manifold, cost=mycost2,verbosity=0)
    #solver = TrustRegions()
    solver = ConjugateGradient()
    #solver = ParticleSwarm()
    #solver = SteepestDescent()
    A_tw = solver.solve(problem)
    return torch.tensor(A_tw)

def bootstraping5(samples,sigma1,sigma2,alpha,num_components,window,change_points,num_perm):
    distances,red_thres = torch.zeros(window*change_points-1),torch.zeros(window*change_points-1)
    ori_distances,ori_thres = torch.zeros(window*change_points-1),torch.zeros(window*change_points-1)
    #initialise
    m,d=samples.shape
    window_size = m/(window*change_points)
    x = samples[0:int(1*window_size),:]
    y = samples[int(1*window_size):int(2*window_size),:]
    #mmd with original data
    ori_dist,ori_thre = kernel_mmd(x,y,sigma1,n_perm=num_perm)

    ori_distances[0],ori_thres[0] = ori_dist,ori_thre
    #dimension reduced mmd
    distance,red_thre = reduced_mmd(x, y, sigma2, alpha,num_components,n_perm=num_perm)
    distances[0], red_thres[0] = distance,red_thre
    for j in range(1,(window*change_points-1)):
        x = samples[int(j*window_size):int((j+1)*window_size),:]
        y = samples[int((j+1)*window_size):int((j+2)*window_size),:]
        #mmd with original data
        ori_dist,ori_thre = kernel_mmd(x,y,sigma1,n_perm=num_perm)

        ori_distances[j],ori_thres[j] = ori_dist,ori_thre
        #dimension reduced mmd
        distance,red_thre = reduced_mmd(x, y, sigma2, alpha,num_components,n_perm=num_perm)
        distances[j], red_thres[j] = distance,red_thre
 
    return ori_distances,ori_thres,distances,red_thres

def generate_laplace(sample_num,dimension,change_points,var,mean,a):
    #data set
    dist_lap = laplace.Laplace(mean[0], var)
    samples = dist_lap.sample((sample_num,dimension))
    for i in range(change_points-1):
        #randomly select a dimensions to change distribution
        changes = torch.randint(0,dimension,(a,))
        new_dist = laplace.Laplace(mean[i+1],var)
        rows = int((i+1)*sample_num/change_points)
        rows1 = int((change_points-i-1)*sample_num/change_points)
        samples[rows:,changes] = new_dist.sample((rows1,a))
    return samples

