#!/usr/bin/env python3
"""
Profiling script for large arrays (1e6-1e7) to find optimization hotspots
"""
import numpy as np
import modmesh
import time

def profile_large_arrays():
    """Profile large arrays to find hotspots for 1e6-1e7 size optimization"""
    
    print("=== Large Arrays Profiling (1e6-1e7) ===\n")
    
    # Test configurations for large arrays
    test_configs = [
        # (size, dtype, description)
        (1000000, 'float32', '1M float32'),
        (1000000, 'float64', '1M float64'),
        (10000000, 'float32', '10M float32'),
        (10000000, 'float64', '10M float64'),
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
        profile_with_callprofiler(nparr, sarr, "contiguous", size)
        
        # Test non-contiguous arrays (stride 2)
        nparr_strided = nparr[::2]
        if dtype == 'float32':
            sarr_strided = modmesh.SimpleArrayFloat32(array=nparr_strided)
        else:
            sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
        print("Non-contiguous arrays (stride 2):")
        profile_with_callprofiler(nparr_strided, sarr_strided, "strided", size//2)
        
        print()

def profile_with_callprofiler(nparr, sarr, array_type, size):
    """Profile using modmesh call profiler"""
    
    # Reset profiler
    modmesh.call_profiler.reset()
    
    # Run multiple iterations for better profiling
    num_iterations = 10 if size >= 10000000 else 100
    
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

def profile_memory_access_patterns():
    """Profile different memory access patterns for large arrays"""
    print("\n=== Memory Access Patterns Analysis ===\n")
    
    # Test different stride patterns
    stride_patterns = [
        (2, "stride 2"),
        (3, "stride 3"),
        (5, "stride 5"),
        (7, "stride 7"),
        (11, "stride 11"),
        (13, "stride 13"),
    ]
    
    size = 1000000
    dtype = 'float64'
    
    print(f"Testing {size} elements with {dtype}")
    
    try:
        # Create base array
        nparr = np.random.rand(size).astype(dtype)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        
        for stride, description in stride_patterns:
            print(f"\n--- {description} ---")
            
            try:
                # Create strided array
                nparr_strided = nparr[::stride]
                sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
                
                # Profile
                profile_single_stride(nparr_strided, sarr_strided, description, stride)
                
            except Exception as e:
                print(f"  ❌ Error at {description}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error in profile_memory_access_patterns: {e}")
        import traceback
        traceback.print_exc()

def profile_single_stride(nparr, sarr, description, stride):
    """Profile a single stride pattern"""
    num_iterations = 100
    
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
    
    print(f"  {description}: SimpleArray {sa_time*1000:.3f}ms vs NumPy {np_time*1000:.3f}ms ({speedup:.2f}x {faster})")

if __name__ == "__main__":
    try:
        profile_large_arrays()
        profile_memory_access_patterns()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
