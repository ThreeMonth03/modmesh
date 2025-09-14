#!/usr/bin/env python3
"""
Test cases for mean() function with non-contiguous arrays
"""
import unittest
import numpy as np
import modmesh


class TestMeanNonContiguous(unittest.TestCase):
    """Test mean() function with various non-contiguous array patterns"""

    def setUp(self):
        """Set up test data"""
        self.tolerance = 1e-10

    def test_1d_strided_array(self):
        """Test 1D strided arrays (e.g., arr[::2])"""
        # Test different stride patterns
        test_cases = [
            (1000, 2),    # every 2nd element
            (1000, 3),    # every 3rd element
            (1000, 5),    # every 5th element
            (1000, 7),    # every 7th element
        ]
        
        for size, step in test_cases:
            with self.subTest(size=size, step=step):
                # Create test data
                nparr = np.arange(size, dtype='float64')
                nparr_strided = nparr[::step]
                sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
                
                # Compute means
                np_mean = np.mean(nparr_strided)
                sarr_mean = sarr_strided.mean()
                
                # Verify results
                self.assertAlmostEqual(np_mean, sarr_mean, places=10,
                                     msg=f"Failed for size={size}, step={step}")
                
                # Verify array properties
                self.assertEqual(sarr_strided.size, len(nparr_strided))
                self.assertFalse(sarr_strided.stride[0] == 1, 
                               "Array should be non-contiguous")

    def test_2d_strided_array(self):
        """Test 2D strided arrays (e.g., arr[::2, ::3])"""
        test_cases = [
            ((100, 100), (2, 3)),    # every 2nd row, every 3rd col
            ((200, 50), (3, 2)),     # every 3rd row, every 2nd col
            ((50, 200), (5, 4)),     # every 5th row, every 4th col
        ]
        
        for (rows, cols), (row_step, col_step) in test_cases:
            with self.subTest(rows=rows, cols=cols, row_step=row_step, col_step=col_step):
                # Create test data
                nparr = np.arange(rows * cols, dtype='float64').reshape(rows, cols)
                nparr_strided = nparr[::row_step, ::col_step]
                sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
                
                # Compute means
                np_mean = np.mean(nparr_strided)
                sarr_mean = sarr_strided.mean()
                
                # Verify results
                self.assertAlmostEqual(np_mean, sarr_mean, places=10,
                                     msg=f"Failed for shape=({rows},{cols}), steps=({row_step},{col_step})")

    def test_3d_strided_array(self):
        """Test 3D strided arrays (e.g., arr[::2, ::3, ::5])"""
        test_cases = [
            ((20, 20, 20), (2, 3, 5)),    # complex 3D striding
            ((30, 15, 10), (3, 2, 2)),    # different strides
            ((10, 30, 20), (5, 3, 4)),    # asymmetric strides
        ]
        
        for (d1, d2, d3), (s1, s2, s3) in test_cases:
            with self.subTest(shape=(d1, d2, d3), strides=(s1, s2, s3)):
                # Create test data
                nparr = np.arange(d1 * d2 * d3, dtype='float64').reshape(d1, d2, d3)
                nparr_strided = nparr[::s1, ::s2, ::s3]
                sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
                
                # Compute means
                np_mean = np.mean(nparr_strided)
                sarr_mean = sarr_strided.mean()
                
                # Verify results
                self.assertAlmostEqual(np_mean, sarr_mean, places=10,
                                     msg=f"Failed for shape=({d1},{d2},{d3}), strides=({s1},{s2},{s3})")

    def test_reverse_strided_array(self):
        """Test reverse strided arrays (e.g., arr[::-1])"""
        test_cases = [
            (1000, -1),    # reverse 1D
            (100, 50, -2), # reverse 2D with step
            (50, 100, -3), # reverse 2D with different step
        ]
        
        for *shape, step in test_cases:
            with self.subTest(shape=shape, step=step):
                # Create test data
                nparr = np.arange(np.prod(shape), dtype='float64').reshape(shape)
                nparr_strided = nparr[::step]
                sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
                
                # Compute means
                np_mean = np.mean(nparr_strided)
                sarr_mean = sarr_strided.mean()
                
                # Verify results
                self.assertAlmostEqual(np_mean, sarr_mean, places=10,
                                     msg=f"Failed for shape={shape}, step={step}")

    def test_mixed_strided_array(self):
        """Test mixed strided arrays with different patterns"""
        test_cases = [
            # (shape, slicing_pattern, description)
            ((100, 100), (slice(10, 90, 2), slice(20, 80, 3)), "2D mixed striding"),
            ((50, 50, 50), (slice(5, 45, 3), slice(10, 40, 2), slice(15, 35, 4)), "3D mixed striding"),
            ((200, 100), (slice(None, None, 2), slice(10, 90, 2)), "2D partial striding"),
        ]
        
        for shape, slicing_pattern, description in test_cases:
            with self.subTest(description=description):
                # Create test data
                nparr = np.arange(np.prod(shape), dtype='float64').reshape(shape)
                nparr_strided = nparr[slicing_pattern]
                sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
                
                # Compute means
                np_mean = np.mean(nparr_strided)
                sarr_mean = sarr_strided.mean()
                
                # Verify results
                self.assertAlmostEqual(np_mean, sarr_mean, places=10,
                                     msg=f"Failed for {description}")

    def test_large_strided_array(self):
        """Test large strided arrays to verify performance"""
        # Large 1D strided array
        nparr = np.arange(1000000, dtype='float64')
        nparr_strided = nparr[::7]  # every 7th element
        sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
        
        # Compute means
        np_mean = np.mean(nparr_strided)
        sarr_mean = sarr_strided.mean()
        
        # Verify results
        self.assertAlmostEqual(np_mean, sarr_mean, places=10,
                             msg="Failed for large strided array")
        
        # Verify array properties
        self.assertEqual(sarr_strided.size, len(nparr_strided))
        self.assertGreater(sarr_strided.stride[0], 1, 
                         "Array should be non-contiguous")

    def test_edge_cases(self):
        """Test edge cases for non-contiguous arrays"""
        # Empty array - create directly with SimpleArray
        sarr_empty = modmesh.SimpleArrayFloat64(0)
        self.assertEqual(sarr_empty.mean(), 0.0)
        
        # Single element
        nparr_single = np.array([42.0])
        sarr_single = modmesh.SimpleArrayFloat64(array=nparr_single)
        self.assertEqual(sarr_single.mean(), 42.0)
        
        # Two elements with stride
        nparr_two = np.array([1.0, 2.0, 3.0, 4.0])[::2]
        sarr_two = modmesh.SimpleArrayFloat64(array=nparr_two)
        self.assertEqual(sarr_two.mean(), 2.0)  # (1.0 + 3.0) / 2

    def test_contiguous_vs_non_contiguous(self):
        """Test that contiguous and non-contiguous arrays give same results"""
        # Create base array
        nparr_base = np.arange(1000, dtype='float64')
        
        # Contiguous array
        sarr_contiguous = modmesh.SimpleArrayFloat64(array=nparr_base)
        
        # Non-contiguous array (every 2nd element)
        nparr_strided = nparr_base[::2]
        sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
        
        # Compute means
        mean_contiguous = sarr_contiguous.mean()
        mean_strided = sarr_strided.mean()
        
        # They should be different (different data)
        self.assertNotEqual(mean_contiguous, mean_strided)
        
        # But strided mean should match numpy
        np_mean = np.mean(nparr_strided)
        self.assertAlmostEqual(mean_strided, np_mean, places=10)

    def test_performance_comparison(self):
        """Test performance comparison between contiguous and non-contiguous"""
        import time
        
        # Create test data
        nparr = np.arange(100000, dtype='float64')
        nparr_strided = nparr[::3]  # every 3rd element
        sarr_strided = modmesh.SimpleArrayFloat64(array=nparr_strided)
        
        # Time SimpleArray mean
        start_time = time.time()
        for _ in range(100):
            _ = sarr_strided.mean()
        sarr_time = (time.time() - start_time) / 100
        
        # Time NumPy mean
        start_time = time.time()
        for _ in range(100):
            _ = np.mean(nparr_strided)
        np_time = (time.time() - start_time) / 100
        
        # Both should be reasonably fast
        self.assertLess(sarr_time, 1.0, "SimpleArray mean too slow")
        self.assertLess(np_time, 1.0, "NumPy mean too slow")
        
        print(f"\nPerformance comparison:")
        print(f"SimpleArray mean: {sarr_time:.6f} seconds")
        print(f"NumPy mean: {np_time:.6f} seconds")
        print(f"Ratio (NumPy/SimpleArray): {np_time/sarr_time:.2f}x")


if __name__ == '__main__':
    unittest.main()
