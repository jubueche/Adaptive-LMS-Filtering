
# coding: utf-8

# In[251]:


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# In[252]:


sysorder = 5
N=2000
h = np.array([0.0976,0.2873,0.3360,0.2210,0.0964])
inp = np.transpose(np.random.rand(1,N))
n = np.random.rand(1,N)
b,a = signal.butter(2,0.25,'low')
Gz = signal.TransferFunction(b,a,dt=0.1)
t_out, y = signal.dlsim(Gz,inp) #Important to use discrete simulation
y = np.concatenate(y)
n = np.concatenate(n * np.std(y)/(10*np.std(n)))
d = y + n
totallength = np.size(d)
N = 60


# In[256]:


w = np.zeros([1,sysorder])
e = np.zeros([1,totallength])
for n in range(sysorder, N):
    index = np.linspace(n,n-sysorder+1,sysorder,dtype='int') #start stop number=stop
    u = np.transpose(np.concatenate(inp[index]))
    y[n] = w.dot(u)
    
    e[0,n] = d[n] - y[n]
    if n < 20:
        mu = 0.32
    else: mu = 0.15
        
    w = w + mu*u*e[0,n]


# In[258]:


for n in range(N+1,totallength):
    index = np.linspace(n,n-sysorder+1,sysorder,dtype='int') #start stop number=stop
    u = np.transpose(np.concatenate(inp[index]))
    y[n] = w.dot(u)
    e[0,n] = d[n] - y[n]


# In[259]:


t = np.arange(totallength)
plt.plot(t,np.transpose(e[:,0:totallength]))
plt.yscale('log')
plt.show()


# In[260]:


plt.plot(t,y)
plt.plot(t,np.transpose(d))
plt.show()


# In[271]:


x = np.linspace(1,np.size(h),np.size(h))
plt.scatter(x, h)
plt.scatter(x, w)
plt.show()

