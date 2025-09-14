#!/usr/bin/env python3
"""
Profiling script for float32/float64 small arrays to find optimization hotspots
"""
import numpy as np
import modmesh
import time

def profile_float_small_arrays():
    """Profile float arrays with small sizes to find hotspots"""
    
    print("=== Float Small Arrays Profiling ===\n")
    
    # Test configurations for small arrays
    test_configs = [
        # (size, dtype, description)
        (10, 'float32', 'Very small float32'),
        (100, 'float32', 'Small float32'),
        (1000, 'float32', 'Medium float32'),
        (10, 'float64', 'Very small float64'),
        (100, 'float64', 'Small float64'),
        (1000, 'float64', 'Medium float64'),
    ]
    
    for size, dtype, description in test_configs:
        print(f"--- {description} (N={size}) ---")
        
        # Create test data
        if dtype == 'float32':
            nparr = np.random.rand(size).astype('float32')
            sarr = modmesh.SimpleArrayFloat32(array=nparr)
        else:  # float64
            nparr = np.random.rand(size).astype('float64')
            sarr = modmesh.SimpleArrayFloat64(array=nparr)
        
        # Test contiguous arrays
        print("Contiguous arrays:")
        profile_with_callprofiler(nparr, sarr, "contiguous")
        
        # Test non-contiguous arrays (stride 2)
        nparr_strided = nparr[::2]
        if dtype == 'float32':
            sarr_strided = modmesh.SimpleArrayFloat32(array=nparr_strided)
        else:
            sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
        print("Non-contiguous arrays (stride 2):")
        profile_with_callprofiler(nparr_strided, sarr_strided, "strided")
        
        print()

def profile_with_callprofiler(nparr, sarr, array_type):
    """Profile using modmesh call profiler"""
    
    # Reset profiler
    modmesh.call_profiler.reset()
    
    # Run multiple iterations for better profiling
    num_iterations = 1000 if len(nparr) < 100 else 100
    
    # Profile SimpleArray mean
    for _ in range(num_iterations):
        _ = sarr.mean()
    
    # Get profiler results
    result = modmesh.call_profiler.result()
    
    # Calculate timing
    start_time = time.time()
    for _ in range(num_iterations):
        _ = sarr.mean()
    sa_time = (time.time() - start_time) / num_iterations
    
    # NumPy timing
    start_time = time.time()
    for _ in range(num_iterations):
        _ = np.mean(nparr)
    np_time = (time.time() - start_time) / num_iterations
    
    print(f"  Array type: {array_type}")
    print(f"  Array size: {sarr.size}")
    print(f"  NumPy time: {np_time*1000:.3f} ms")
    print(f"  SimpleArray time: {sa_time*1000:.3f} ms")
    print(f"  Speedup: {np_time/sa_time:.2f}x")
    
    # Print detailed profiler breakdown
    print("  Profiler breakdown:")
    print_profiler_tree(result, 2)

def print_profiler_tree(data, indent=0):
    """Print profiler results in tree format"""
    spaces = "  " * indent
    
    if isinstance(data, dict):
        if 'name' in data and 'total_time' in data:
            name = data['name']
            total_time = data['total_time']
            count = data.get('count', 1)
            avg_time = total_time / count if count > 0 else 0
            
            print(f"{spaces}{name}: {total_time:.6f}s total, {avg_time:.6f}s avg (count: {count})")
            
        if 'children' in data and data['children']:
            for child in data['children']:
                print_profiler_tree(child, indent + 1)
    elif isinstance(data, list):
        for item in data:
            print_profiler_tree(item, indent)

def profile_optimization_opportunities():
    """Look for specific optimization opportunities"""
    print("\n=== Optimization Opportunities Analysis ===\n")
    
    # Test very small arrays where NumPy might still be faster
    sizes = [10, 50, 100, 500, 1000]
    
    for size in sizes:
        print(f"--- Array size: {size} ---")
        
        # Test float32
        nparr32 = np.random.rand(size).astype('float32')
        sarr32 = modmesh.SimpleArrayFloat32(array=nparr32)
        
        # Test float64
        nparr64 = np.random.rand(size).astype('float64')
        sarr64 = modmesh.SimpleArrayFloat64(array=nparr64)
        
        # Profile both
        profile_single_size(nparr32, sarr32, "float32", size)
        profile_single_size(nparr64, sarr64, "float64", size)
        print()

def profile_single_size(nparr, sarr, dtype, size):
    """Profile a single array size"""
    num_iterations = 10000 if size < 100 else 1000
    
    # SimpleArray timing
    start_time = time.time()
    for _ in range(num_iterations):
        _ = sarr.mean()
    sa_time = (time.time() - start_time) / num_iterations
    
    # NumPy timing
    start_time = time.time()
    for _ in range(num_iterations):
        _ = np.mean(nparr)
    np_time = (time.time() - start_time) / num_iterations
    
    ratio = sa_time / np_time
    faster = "faster" if ratio < 1 else "slower"
    speedup = 1/ratio if ratio < 1 else ratio
    
    print(f"  {dtype}: SimpleArray {sa_time*1000:.3f}ms vs NumPy {np_time*1000:.3f}ms ({speedup:.2f}x {faster})")

if __name__ == "__main__":
    profile_float_small_arrays()
    profile_optimization_opportunities()
