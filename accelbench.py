# Adding code to enable multithreading at the very beginning
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ["KMP_INIT_AT_FORK"] = "FALSE"


import platform

# Determine the number of CPU cores
num_cores = os.cpu_count() or 1  # In case os.cpu_count() returns None
# Remove the limitation on core usage if Windows imposed it
if hasattr(os, 'sched_setaffinity'):
    os.sched_setaffinity(0, list(range(os.cpu_count())))

# Set environment variables for MKL/OpenBLAS
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)

os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["MKL_THREADING_LAYER"] = "GNU"

# Now import the remaining libraries
import numpy as np
import torch
import time
import sys
import warnings
from collections import namedtuple
from threadpoolctl import threadpool_limits

warnings.filterwarnings('ignore')

# Set the number of threads for Numpy (additional guarantee)
if hasattr(np, 'set_num_threads'):
    np.set_num_threads(num_cores)


# Define the configuration structure
TestConfig = namedtuple('TestConfig', [
    'name',           # Human-readable type name
    'numpy_dtype',    # Data type for NumPy (None if not supported)
    'torch_dtype',    # Data type for PyTorch
    'devices',        # List of devices for PyTorch testing ['cpu', 'cuda']
    'repeats',        # Number of repeats for warm-up
    'skip_numpy'      # Flag to skip NumPy tests (even if type is supported)
])

# Benchmark parameters
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

print(f'Starting benchmark. Multiply two matrices with [{N} x {N}] elements and sum() the result.')
print(f"Using {num_cores} CPU cores for NumPy computations")

# Calculation of operations for TFLOPS
totalmul = (N * N * N) + ((N - 1) * N * N)  # N^3 mult + (N-1)*N^2 additions
totalsum = N * N                            # Sum of all elements
totalop = totalmul + totalsum               # Total operations
NS_TO_SEC = 1e-9                            # Nanoseconds to seconds converter

# Data generation (stored in float64 for maximum accuracy of source data)
np.random.seed(42)
torch.manual_seed(42)
x_base = 0.2 * np.random.rand(N, N) - 0.1  # Values in the range [-0.1, +0.1]
y_base = 0.2 * np.random.rand(N, N) - 0.1

# Definition of available devices
available_devices = ['cpu']
if torch.cuda.is_available():
    available_devices.append('cuda:0')

# Test configurations
TEST_CONFIGS = [
    TestConfig(
        name='float16',
        numpy_dtype=np.float16,
        torch_dtype=torch.float16,
        devices=['cuda:0'],  # float16 on CPU is often slower than float32
        repeats=10,
        skip_numpy=True    # Skip NumPy for float16 on x86
    ),
    TestConfig(
        name='float32',
        numpy_dtype=np.float32,
        torch_dtype=torch.float32,
        devices=['cuda:0'],
        repeats=6,
        skip_numpy=False
    ),
    TestConfig(
        name='float64',
        numpy_dtype=np.float64,
        torch_dtype=torch.float64,
        devices=['cuda:0'],   # float64 on GPU is usually not optimized
        repeats=2,
        skip_numpy=False
    ),
    TestConfig(
        name='bfloat16',
        numpy_dtype=None,   # Not supported in NumPy
        torch_dtype=torch.bfloat16,
        devices=['cuda:0'],
        repeats=10,
        skip_numpy=True
    )
]

def numpy_benchmark(x, y, config):
    """Execute benchmark on NumPy"""
    if config.numpy_dtype is None or config.skip_numpy:
        print(f"        SKIPPED NumPy for {config.name} (not supported or skipped)")
        return None, None
    
    print(f"        [NumPy] dtype={config.numpy_dtype.__name__}, N={N}")
    
    # Convert data to the target type
    x_conv = x.astype(config.numpy_dtype)
    y_conv = y.astype(config.numpy_dtype)
    
    best_time = float('inf')
    result = None
    
    # Warm-up and time measurement
    for _ in range(3):
        start = time.perf_counter_ns()
        with threadpool_limits(limits=24, user_api='blas'):
            z = np.matmul(x_conv, y_conv)
            result = np.sum(z)

        elapsed = time.perf_counter_ns() - start
        
        if elapsed < best_time:
            best_time = elapsed
    
    return result, best_time

def pytorch_benchmark(x, y, config, device_str):
    """Execute benchmark on PyTorch for the specified device"""
    device = torch.device(device_str)
    print(f"        [PyTorch] device={device_str}, dtype={config.torch_dtype}, N={N}")
    
    # Special handling for bfloat16 - intermediate conversion to float32 is required
    if config.torch_dtype == torch.bfloat16:
        x_conv = x.astype(np.float32)
        y_conv = y.astype(np.float32)
    else:
        x_conv = x
        y_conv = y
    
    # Creating tensors on the target device
    x_tensor = torch.tensor(x_conv, dtype=config.torch_dtype, device=device)
    y_tensor = torch.tensor(y_conv, dtype=config.torch_dtype, device=device)
    
    best_time = float('inf')
    result = None
    
    # Repeats for warm-up and selecting the best time
    for _ in range(config.repeats):
        if device_str == 'cuda':
            torch.cuda.synchronize(device)
        
        start = time.perf_counter_ns()
        z = torch.mm(x_tensor, y_tensor)
        result = torch.sum(z)
        
        if device_str == 'cuda':
            torch.cuda.synchronize(device)
        else:
            # Forcing completion of computations on CPU
            result.item()
        
        elapsed = time.perf_counter_ns() - start
        if elapsed < best_time:
            best_time = elapsed
    
    # Clear GPU memory
    if device_str == 'cuda':
        del x_tensor, y_tensor, z
        torch.cuda.empty_cache()
    
    return result.item(), best_time

def calculate_tflops(time_ns):
    """Calculate performance in TFLOPS"""
    if time_ns <= 0:
        return 0.0
    return totalop / (time_ns * NS_TO_SEC) / 1e12

def format_value(value, float_format='.6e'):
    """Universal value formatting for output"""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return f"{value:{float_format}}"
    return str(value)

def format_time(time_val):
    """Format execution time"""
    if isinstance(time_val, str):
        return time_val
    return f"{time_val:.4f}"

def format_tflops(tflops_val):
    """Format performance in TFLOPS"""
    if isinstance(tflops_val, str):
        return tflops_val
    return f"{tflops_val:.3f}"

# Main test loop
results = []
print(f"{'='*80}")
print(f"GPU/CPU Benchmark (N={N})")
print(f"{'='*80}")

for config in TEST_CONFIGS:
    print(f"\n[TESTING TYPE: {config.name}]")
    print(f"{'-'*60}")
    
    # Testing NumPy (only on CPU)
    if not config.skip_numpy and config.numpy_dtype is not None:
        try:
            res, duration = numpy_benchmark(x_base, y_base, config)
            if duration is not None:
                tflops = calculate_tflops(duration)
                results.append({
                    'framework': 'NumPy',
                    'dtype': config.name,
                    'device': 'cpu',
                    'N': N,
                    'result': res,
                    'time_sec': duration * NS_TO_SEC,
                    'tflops': tflops
                })
        except Exception as e:
            print(f"        NumPy test failed: {str(e)}")
            results.append({
                'framework': 'NumPy',
                'dtype': config.name,
                'device': 'cpu',
                'N': N,
                'result': 'ERROR',
                'time_sec': '-',
                'tflops': '-'
            })
    else:
        print(f"        SKIPPED NumPy tests for {config.name}")
        results.append({
            'framework': 'NumPy',
            'dtype': config.name,
            'device': 'cpu',
            'N': N,
            'result': '-',
            'time_sec': '-',
            'tflops': '-'
        })
    
    # Testing PyTorch on all available devices from the configuration
    for device_str in config.devices:
        if device_str not in available_devices:
            print(f"        SKIPPED {device_str} for {config.name} (device not available)")
            results.append({
                'framework': 'PyTorch',
                'dtype': config.name,
                'device': device_str,
                'N': N,
                'result': f"SKIPPED ({device_str} not available)",
                'time_sec': '-',
                'tflops': '-'
            })
            continue
        
        try:
            res, duration = pytorch_benchmark(x_base, y_base, config, device_str)
            tflops = calculate_tflops(duration)
            results.append({
                'framework': 'PyTorch',
                'dtype': config.name,
                'device': device_str,
                'N': N,
                'result': res,
                'time_sec': duration * NS_TO_SEC,
                'tflops': tflops
            })
        except Exception as e:
            print(f"        PyTorch {device_str} test failed: {str(e)}")
            results.append({
                'framework': 'PyTorch',
                'dtype': config.name,
                'device': device_str,
                'N': N,
                'result': 'ERROR',
                'time_sec': '-',
                'tflops': '-'
            })

# Output of results
print(f"\n{'='*110}")
header = f"{'Framework':<10} | {'Data Type':<10} | {'Device':<7} | {'N':<6} | {'Result (sum)':<20} | {'Time (sec)':<10} | {'TFLOPS':<10}"
print(header)

for i, res in enumerate(results):
    # Preparing formatted values
    result_val = format_value(res['result'])
    time_val = format_time(res['time_sec'])
    tflops_val = format_tflops(res['tflops'])
    
    # Forming the result string
    row = (
        f"{res['framework']:<10} | "
        f"{res['dtype']:<10} | "
        f"{res['device']:<7} | "
        f"{res['N']:<6} | "
        f"{result_val:<20} | "
        f"{time_val:<10} | "
        f"{tflops_val:<10}"
    )
    if i%2==0:
        print(f"{'-'*110}") 
    print(row)

# System information

# Operating system
os_name = platform.system()              # For example: 'Linux', 'Windows', 'Darwin' (macOS)
os_release = platform.release()          # OS version
os_version = platform.version()          # Additional version information

# Processor (CPU)
cpu_info = platform.processor()          # Processor name (may be empty on some systems)

# Alternative way to get more reliable CPU information (especially on Linux/macOS)

if os_name == "Linux":
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_info = line.split(":")[1].strip()
                    break
    except Exception:
        pass
elif os_name == "Darwin":  # macOS
    try:
        cpu_info = os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
    except Exception:
        pass
elif os_name == "Windows":
    # On Windows platform.processor() usually works fine,
    # but you can also use WMIC (if available):
    try:
        import subprocess
        result = subprocess.run(
            ["wmic", "cpu", "get", "name"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            cpu_info = lines[1].strip()
    except Exception:
        pass

# Output

print(f"\n{'='*110}")
print("System Configuration:")
print(f"OS: {os_name} {os_release} ({os_version})")
print(f"CPU: {cpu_info or 'Could not determine'}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Available devices: {', '.join(available_devices)}")
if 'cuda:0' in available_devices:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"{'='*110}")
