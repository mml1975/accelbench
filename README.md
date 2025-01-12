# AccelBench
### Benchmark CPU and GPU in TFLOPs by multiple matrix with Numpy and PyTorch



Run example for C = A[N,N] x B[N,N] and sum (C). N - size = 10000:\
$ python accelbench.py 10000\
Starting with Type = <class 'numpy.float16'>\
        PyTorch cuda part. device = cuda  N = 10000\
Starting with Type = <class 'numpy.float32'>\
        Numpy part. N = 10000\
        PyTorch cuda part. device = cuda  N = 10000\
Starting with Type = <class 'numpy.float64'>\
        Numpy part. N = 10000\
        PyTorch cuda part. device = cuda  N = 10000

Name	| bit	| device| N	|           sum            |  sec     | TFLOPS   \
--------+-------+-------+-------+--------------------------+----------+--------\
Numpy	| 16	| cpu	| 10000	|  -                       | -        | -\
PyTorch	| 16	| cuda	| 10000	|                     2260 |   0.0237 | 84.4	\
Numpy	| 32	| cpu	| 10000	|           2261.412109375 |     1.54 | 1.3	\
PyTorch	| 32	| cuda	| 10000	|        2261.409912109375 |   0.0979 | 20.4	\
Numpy	| 64	| cpu	| 10000	|   2261.41170710232563579 |     3.09 | 0.648	\
PyTorch	| 64	| cuda	| 10000	|   2261.41170710235383012 |     2.09 | 0.959\
