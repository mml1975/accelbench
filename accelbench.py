import numpy as np
import torch
import time
import sys
import warnings
warnings.filterwarnings('ignore') 

#N=17000 # Matrix size [N,N]
N = int(sys.argv[1])

totalmul = (N*N*N) + ((N-1)*N*N) # multiple matrixes: N^3 mult + (N-1)*N- additions
totalsum = (N*N) # N for sum of the all elements
totalop = totalmul + totalsum # Total operations
nsec = 1/1000000000. # billion nanosec's
# for scaling perf_counter_ns to sec's


np.random.seed(42)
x1 = 0.2*np.random.rand(N,N) - 0.1  # Must be in [-0.1 +0.1]
y1 = 0.2*np.random.rand(N,N) - 0.1

dtypes = [(np.float16, '16',10),(np.float32, '32',6),(np.float64, '64',2)]
# structure: data types, string title, repeatness for "heating" gpu


def numpytest(x,y, qnt):
   # Calculate with Numpy on the CPU
   # x,y - source arrays, qnt - size  x,y[qnt,qnt]
   # return tuple of the sum of all elements (z=x*y) and time for it in a perf_counter_ns
   
   print(f"        Numpy part. N = {qnt}")  
   start_t = time.perf_counter_ns()
   
   z = np.matmul(x,y)
   t = np.sum(z)
   
   end_t = time.perf_counter_ns()
   return (t,end_t-start_t)
   
def torchcudatest(x,y,qnt,nrepeat,device):
    # Calculate with pytorch on 'cuda'
    print(f"        PyTorch cuda part. device = {device}  N = {qnt}")
    
    a = torch.from_numpy(x).to(device)
    b = torch.from_numpy(y).to(device)
    
    best_time = 1e+12
    for i in range(nrepeat):
        torch.cuda.synchronize(device = device)
        start_t = time.perf_counter_ns()
        c = torch.mm(a,b) # automatically on cuda
        t = torch.sum(c)
        torch.cuda.synchronize(device = device) # Otherwise end_t calculating before torch
        end_t = time.perf_counter_ns()
        dt = end_t-start_t
        if dt < best_time:
            best_time = dt
        del c
    
    #free memory_allocated
    del a; del b;
    torch.cuda.empty_cache()
    return(t,best_time)
    
def torchcputest(x,y,qnt,nrepeat,device):
    # Calculate with pytorch on 'cuda' (TPU)
    print(f"       PyTorch part. device = CPU  N = {qnt}")
    
    a = torch.from_numpy(x)
    b = torch.from_numpy(y)
    
    torch.cpu.synchronize(device = device)
    start_t = time.perf_counter_ns()
    c = torch.mm(a,b)
    t = torch.sum(c)
    torch.cpu.synchronize(device = device) 
    end_t = time.perf_counter_ns()
    
    #free memory_allocated
    del a; del b; del c
    return(t,end_t-start_t)
    
def gettflops(timedelta):
    return totalop / (nsec*timedelta) / 1000 /1000 / 1000 / 1000


results = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torchfunc = {'cuda':torchcudatest, 'cpu':torchcputest}


# Main bench
for type_item in dtypes:
   print(f"Starting with Type = {type_item[0]}")
   x = np.ndarray.astype(x1,type_item[0]) # Convert values ​​to the selected type
   y = np.ndarray.astype(y1,type_item[0])  
     
   if type_item[0]!=np.float16: # fp16 too slow on the x86, skip them
       (t,calctime) = numpytest(x,y,N);tflops = gettflops(calctime)
       s = f"Numpy\t| {type_item[1]}\t| cpu\t| {N}\t| {t:24.21g} | {nsec*calctime:8.3g} | {tflops:.3g}\t"
       results.append(s)
   else:
       s = f"Numpy\t| {type_item[1]}\t| cpu\t| {N}\t|  -                       | -        | -"
       results.append(s)

   (t,calctime) = torchfunc[device](x,y,N,type_item[2],torch.device(device))
   tflops = gettflops(calctime)
   s = f"PyTorch\t| {type_item[1]}\t| {device}\t| {N}\t| {t:24.21g} | {nsec*calctime:8.3g} | {tflops:.3g}\t"
   results.append(s)
    
print()
print("Name\t| bit\t| device| N\t|           sum            |  sec     | TFLOPS   ")
print("--------+-------+-------+-------+--------------------------+----------+--------")
for s in results:
    print(s)
