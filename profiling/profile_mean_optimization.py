#!/usr/bin/env python3
"""
Profiling script for mean() function optimization
"""
import numpy as np
import modmesh
import time

def profile_mean_performance():
    """Profile mean() function with different array types and sizes"""
    
    # Test configurations
    test_configs = [
        # (size, dtype, description)
        (1000, 'float64', 'Small float64'),
        (100000, 'float64', 'Medium float64'),
        (1000000, 'float64', 'Large float64'),
        (1000, 'float32', 'Small float32'),
        (100000, 'float32', 'Medium float32'),
        (1000000, 'float32', 'Large float32'),
        (1000, 'int64', 'Small int64'),
        (100000, 'int64', 'Medium int64'),
        (1000000, 'int64', 'Large int64'),
    ]
    
    print("=== Mean Performance Profiling ===\n")
    
    for size, dtype, description in test_configs:
        print(f"--- {description} (N={size}) ---")
        
        # Create test data
        if dtype == 'float64':
            nparr = np.random.rand(size).astype('float64')
            sarr = modmesh.SimpleArrayFloat64(array=nparr)
        elif dtype == 'float32':
            nparr = np.random.rand(size).astype('float32')
            sarr = modmesh.SimpleArrayFloat32(array=nparr)
        elif dtype == 'int64':
            nparr = np.random.randint(-1000, 1000, size, dtype='int64')
            sarr = modmesh.SimpleArrayInt64(array=nparr)
        
        # Test contiguous arrays
        print("Contiguous arrays:")
        profile_single_test(nparr, sarr, "contiguous")
        
        # Test non-contiguous arrays (stride 2)
        nparr_strided = nparr[::2]
        if dtype == 'float64':
            sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
        elif dtype == 'float32':
            sarr_strided = modmesh.SimpleArrayFloat32(array=nparr_strided)
        elif dtype == 'int64':
            sarr_strided = modmesh.SimpleArrayInt64(array=nparr_strided)
        print("Non-contiguous arrays (stride 2):")
        profile_single_test(nparr_strided, sarr_strided, "strided")
        
        print()

def profile_single_test(nparr, sarr, test_type):
    """Profile a single test case"""
    num_iterations = 100 if len(nparr) < 100000 else 10
    
    # NumPy timing
    start_time = time.time()
    for _ in range(num_iterations):
        np_result = np.mean(nparr)
    np_time = (time.time() - start_time) / num_iterations
    
    # SimpleArray timing
    start_time = time.time()
    for _ in range(num_iterations):
        sa_result = sarr.mean()
    sa_time = (time.time() - start_time) / num_iterations
    
    # Verify correctness
    if not np.isclose(np_result, sa_result, rtol=1e-10):
        print(f"  WARNING: Results don't match! NP={np_result}, SA={sa_result}")
    
    # Performance comparison
    ratio = sa_time / np_time
    faster = "faster" if ratio < 1 else "slower"
    speedup = 1/ratio if ratio < 1 else ratio
    
    print(f"  NumPy:     {np_time*1000:.3f} ms")
    print(f"  SimpleArray: {sa_time*1000:.3f} ms")
    print(f"  SimpleArray is {speedup:.2f}x {faster} than NumPy")
    
    # Array properties
    print(f"  Array size: {sarr.size}")
    print(f"  Array shape: {sarr.shape}")
    print(f"  Array stride: {sarr.stride}")
    print(f"  Is contiguous: {sarr.stride[-1] == 1 if len(sarr.stride) > 0 else True}")

def profile_with_callprofiler():
    """Profile using modmesh's call profiler"""
    print("\n=== Call Profiler Analysis ===\n")
    
    # Create large test array
    size = 1000000
    nparr = np.random.rand(size).astype('float64')
    sarr = modmesh.SimpleArrayFloat64(array=nparr)
    
    # Reset profiler
    modmesh.call_profiler.reset()
    
    # Profile SimpleArray mean
    print("Profiling SimpleArray mean()...")
    for _ in range(10):
        _ = sarr.mean()
    
    # Get profiler results
    result = modmesh.call_profiler.result()
    print("Call profiler results:")
    print_modmesh_profile(result, 0)

def print_modmesh_profile(data, indent=0):
    """Print modmesh profiler results in a readable format"""
    spaces = "  " * indent
    
    if isinstance(data, dict):
        if 'name' in data and 'total_time' in data:
            print(f"{spaces}{data['name']}: {data['total_time']:.6f}s (count: {data.get('count', 1)})")
            
        if 'children' in data and data['children']:
            for child in data['children']:
                print_modmesh_profile(child, indent + 1)
    elif isinstance(data, list):
        for item in data:
            print_modmesh_profile(item, indent)

if __name__ == "__main__":
    profile_mean_performance()
    profile_with_callprofiler()
