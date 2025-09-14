#!/usr/bin/env python3
"""
Detailed profiling script for mean() function optimization
"""
import numpy as np
import modmesh
import time

def profile_mean_detailed():
    """Detailed profiling of mean() function"""
    
    print("=== Detailed Mean Profiling ===\n")
    
    # Test with large non-contiguous array
    size = 1000000
    nparr = np.random.rand(size).astype('float64')
    nparr_strided = nparr[::2]  # Non-contiguous
    sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
    
    print(f"Array size: {sarr_strided.size}")
    print(f"Array shape: {sarr_strided.shape}")
    print(f"Array stride: {sarr_strided.stride}")
    print(f"Is contiguous: {sarr_strided.stride[-1] == 1}")
    print()
    
    # Profile with call profiler
    print("Profiling with call profiler...")
    modmesh.call_profiler.reset()
    
    # Run multiple iterations
    for i in range(5):
        result = sarr_strided.mean()
        print(f"Iteration {i+1}: result = {result}")
    
    # Get detailed profiler results
    result = modmesh.call_profiler.result()
    print("\nDetailed profiler results:")
    print_profiler_tree(result, 0)
    
    # Compare with NumPy
    print("\nNumPy comparison:")
    start_time = time.time()
    for _ in range(5):
        np_result = np.mean(nparr_strided)
    np_time = (time.time() - start_time) / 5
    
    print(f"NumPy mean: {np_result:.10f}")
    print(f"NumPy time: {np_time*1000:.3f} ms per call")

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

def profile_contiguous_vs_strided():
    """Compare contiguous vs strided performance"""
    print("\n=== Contiguous vs Strided Comparison ===\n")
    
    size = 1000000
    nparr = np.random.rand(size).astype('float64')
    
    # Contiguous array
    sarr_contiguous = modmesh.SimpleArrayFloat64(array=nparr)
    
    # Strided array
    nparr_strided = nparr[::2]
    sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
    
    print("Contiguous array:")
    profile_single_array(sarr_contiguous, "contiguous")
    
    print("\nStrided array:")
    profile_single_array(sarr_strided, "strided")

def profile_single_array(sarr, array_type):
    """Profile a single array"""
    modmesh.call_profiler.reset()
    
    # Run mean() multiple times
    for _ in range(10):
        result = sarr.mean()
    
    # Get profiler results
    result = modmesh.call_profiler.result()
    
    print(f"  Array type: {array_type}")
    print(f"  Array size: {sarr.size}")
    print(f"  Array stride: {sarr.stride}")
    print(f"  Mean result: {result}")
    
    # Print profiler tree
    print("  Profiler breakdown:")
    print_profiler_tree(result, 2)

if __name__ == "__main__":
    profile_mean_detailed()
    profile_contiguous_vs_strided()
