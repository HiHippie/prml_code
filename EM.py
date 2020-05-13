import numpy as np
import scipy.stats as sts
from tqdm import tqdm

class Expectation_Maximization:
    def __init__(self, dim, iters=10000, epsilon=1e-6, prior=None):

        self.dim = dim
        self.iters = iters
        self.epsilon = epsilon

        self.prior = prior

        self.A = []
        self.Count = {'B': {'Head': 0, 'Tail': 0}, 'C': {'Head':0, 'Tail':0}}

    def fit(self, Y):

        for iter in range(self.iters):

            self.E_step(Y)

            new_prior = self.M_step()

            params_change = np.abs(self.prior[1] - new_prior[1])

            if params_change < self.epsilon:
                break
            else:
                self.prior = new_prior

        return new_prior, iter

    def E_step(self, Y):

        for observation in Y:
            l = len(observation)
            head_number = np.sum(observation)
            tail_number = l - head_number
            theta_A = self.prior[0]
            theta_B = self.prior[1]
            theta_C = self.prior[2]
            # for B
            contribution_B = theta_A * sts.binom.pmf(head_number, l, theta_B)
            contribution_C = (1 - theta_A) * sts.binom.pmf(head_number, l, theta_C)
            mu_B = contribution_B / (contribution_B + contribution_C)
            mu_C = 1. - mu_B
            # mu_C = contribution_C / (contribution_B + contribution_C)

            self.A.append(mu_B)
            self.Count['B']['Head'] += mu_B * head_number
            self.Count['B']['Tail'] += mu_B * tail_number
            self.Count['C']['Head'] += mu_C * head_number
            self.Count['C']['Tail'] += mu_C * tail_number

    def M_step(self):
        new_A = np.sum(self.A) / len(self.A)
        new_B = self.Count['B']['Head'] / (self.Count['B']['Head'] + self.Count['B']['Tail'])
        new_C = self.Count['C']['Head'] / (self.Count['C']['Head'] + self.Count['C']['Tail'])
        return np.array([new_A, new_B, new_C])


'''
三硬币模型
Y: observation variables  1 for head, 0 for tail
Z: hidden variables [A,B,C]
'''
Y = np.array([[1,0,0,1,0,0,0,1,0,1],
              [1,1,1,0,0,0,1,0,0,0],
              [1,1,0,0,0,0,1,0,1,0],
              [0,0,1,0,1,1,0,1,0,1],
              [1,1,0,0,0,0,1,0,0,0],
              [0,1,1,0,1,0,1,0,1,0],
              [0,0,0,1,1,1,1,1,0,1],
              [0,0,0,0,0,1,1,1,0,0],
              [1,1,0,0,0,0,1,0,0,1],
              [1,1,1,0,0,0,1,0,1,0],
              ])



prior = np.array([0.8, 0.6, 0.5])

EM = Expectation_Maximization(3, prior=prior)
results = EM.fit(Y)
print(results)