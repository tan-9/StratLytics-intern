#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


def bernoulli(n_b, p_b, size_b):
    x = random.binomial(n = n_b, p =p_b, size=size_b)
    p_success = sum(x) / (len(x) * n_b)
    sns.distplot(x, hist=True, kde=False)
    plt.show()
    return {'datapoints':x, 'probability':p_success}

#for bernoulli n=1
#calculating probability for 1 coin toss
print(bernoulli(1, 0.5, 25))

#for binomial n>1
#calculating probability for getting answers right on a mcq test
print(bernoulli(20,0.25,500))


# In[3]:


def poisson(l):
    x = random.poisson(lam=l, size=500)
    sns.distplot(x,hist=True, kde=False)
    plt.show()
poisson(5)


# In[4]:


def uniform(low, high, size):
    x = np.random.uniform(low=low, high=high, size=size)
    sns.distplot(x,hist=False)
    plt.show()
    return x

uniform(5,10, 100)
    


# In[5]:


def normal(mu, sigma, size):
    x = np.random.normal(loc=mu, scale=sigma, size=size)
    sns.distplot(x, hist=False)
    plt.show()
    return x

normal(0,1,2)


# In[13]:


def std_normal(size):
    x = np.random.standard_normal(size)
    sns.distplot(x, hist=False)
    plt.show()
    return x[:10]
std_normal(10000)


# In[6]:


def exponential(inv, size):
    x = np.random.exponential(scale=inv, size=size)
    sns.distplot(x, hist=False)
    plt.show()
    return x[:10]

exponential(0.25, 1000)


# In[7]:


def chi_sq(dof, size):
    x = np.random.chisquare(df=dof, size=size)
    sns.distplot(x, hist=False)
    plt.show()
    return x

chi_sq(3, 100)


# In[ ]:




