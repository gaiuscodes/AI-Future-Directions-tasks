"""
Benchmark script to compare sorting implementations.
"""

import time
import random
from sort_dict_list import sort_by_key_ai_suggested, sort_by_key_manual_copy


def generate_test_data(size):
    """Generate random test data."""
    return [
        {
            "name": f"Person_{i}",
            "age": random.randint(18, 80),
            "score": random.randint(0, 100)
        }
        for i in range(size)
    ]


def benchmark(func, data, key, iterations=100):
    """Benchmark a sorting function."""
    times = []
    for _ in range(iterations):
        test_data = data.copy()
        start = time.perf_counter()
        func(test_data, key)
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times)


if __name__ == "__main__":
    print("Benchmarking sorting implementations...")
    print("=" * 50)
    
    sizes = [100, 1000, 10000]
    iterations = 50
    
    for size in sizes:
        print(f"\nTesting with {size} dictionaries:")
        test_data = generate_test_data(size)
        
        # Benchmark AI-suggested approach
        ai_time = benchmark(sort_by_key_ai_suggested, test_data, "age", iterations)
        
        # Benchmark manual implementation
        manual_time = benchmark(sort_by_key_manual_copy, test_data, "age", iterations)
        
        print(f"  AI-suggested (lambda):     {ai_time*1000:.4f} ms")
        print(f"  Manual (itemgetter):       {manual_time*1000:.4f} ms")
        print(f"  Speedup:                   {ai_time/manual_time:.2f}x")
        print(f"  Improvement:               {(1 - manual_time/ai_time)*100:.1f}% faster")

