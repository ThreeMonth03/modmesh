#!/usr/bin/env python3
"""
Detailed profiling script for large arrays (1e7) to find optimization hotspots
"""
import numpy as np
import modmesh
import time
import sys

def profile_large_arrays_detailed():
    """Profile large arrays to find hotspots for 1e7 size optimization"""
    
    print("=== Large Arrays Detailed Profiling (1e7) ===\n")
    
    # Test configurations for very large arrays
    test_configs = [
        # (size, dtype, description)
        (10000000, 'float32', '10M float32'),
        (10000000, 'float64', '10M float64'),
        # Temporarily disable 50M tests to avoid segmentation fault
        # (50000000, 'float32', '50M float32'),
        # (50000000, 'float64', '50M float64'),
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
        
        # Test different stride patterns
        stride_patterns = [
            (1, "contiguous"),
            (2, "stride 2"),
            (3, "stride 3"),
            (5, "stride 5"),
            (7, "stride 7"),
            (11, "stride 11"),
            (13, "stride 13"),
            (17, "stride 17"),
            (23, "stride 23"),
            (29, "stride 29"),
        ]
        
        for stride, stride_desc in stride_patterns:
            if stride == 1:
                # Contiguous case
                nparr_test = nparr
                sarr_test = sarr
                actual_size = size
            else:
                # Strided case
                nparr_test = nparr[::stride]
                if dtype == 'float32':
                    sarr_test = modmesh.SimpleArrayFloat32(array=nparr_test)
                else:
                    sarr_test = modmesh.SimpleArrayFloat64(array=nparr_test)
                actual_size = len(nparr_test)
            
            print(f"  {stride_desc} (N={actual_size}):")
            
            # Profile SimpleArray
            num_iterations = max(1, 1000000 // actual_size)  # Adjust iterations based on size
            
            # Reset profiler
            modmesh.call_profiler.reset()
            
            # Profile SimpleArray mean
            start_time = time.time()
            for _ in range(num_iterations):
                _ = sarr_test.mean()
            sa_time = (time.time() - start_time) / num_iterations
            
            # NumPy timing
            start_time = time.time()
            for _ in range(num_iterations):
                _ = np.mean(nparr_test)
            np_time = (time.time() - start_time) / num_iterations
            
            ratio = sa_time / np_time
            faster = "faster" if ratio < 1 else "slower"
            speedup = 1/ratio if ratio < 1 else ratio
            
            print(f"    NumPy: {np_time*1000:.3f}ms, SimpleArray: {sa_time*1000:.3f}ms ({speedup:.2f}x {faster})")
            
            # Get profiler results for SimpleArray
            result = modmesh.call_profiler.result()
            print_profiler_breakdown(result, 4)
            print()

def print_profiler_breakdown(data, indent=0):
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
                print_profiler_breakdown(child, indent + 1)
    elif isinstance(data, list):
        for item in data:
            print_profiler_breakdown(item, indent)

def profile_memory_bandwidth():
    """Profile memory bandwidth for different access patterns"""
    print("\n=== Memory Bandwidth Analysis ===\n")
    
    size = 10000000
    dtype = 'float64'
    
    print(f"Testing {size} elements with {dtype}")
    
    # Create base array
    nparr = np.random.rand(size).astype(dtype)
    sarr = modmesh.SimpleArrayFloat64(array=nparr)
    
    # Test different stride patterns
    stride_patterns = [1, 2, 3, 5, 7, 11, 13, 17, 23, 29, 31, 37, 41, 43, 47]
    
    print("Stride | NumPy (ms) | SimpleArray (ms) | Ratio | Status")
    print("-------|------------|------------------|-------|--------")
    
    for stride in stride_patterns:
        if stride == 1:
            nparr_test = nparr
            sarr_test = sarr
        else:
            nparr_test = nparr[::stride]
            sarr_test = modmesh.SimpleArrayFloat64(array=nparr_test)
        
        num_iterations = max(1, 1000000 // len(nparr_test))
        
        # NumPy timing
        start_time = time.time()
        for _ in range(num_iterations):
            _ = np.mean(nparr_test)
        np_time = (time.time() - start_time) / num_iterations
        
        # SimpleArray timing
        start_time = time.time()
        for _ in range(num_iterations):
            _ = sarr_test.mean()
        sa_time = (time.time() - start_time) / num_iterations
        
        ratio = sa_time / np_time
        status = "✅" if ratio < 1 else "❌"
        
        print(f"{stride:6d} | {np_time*1000:10.3f} | {sa_time*1000:15.3f} | {ratio:5.2f} | {status}")

if __name__ == "__main__":
    profile_large_arrays_detailed()
    profile_memory_bandwidth()
