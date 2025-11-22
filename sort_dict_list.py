"""
Sort a list of dictionaries by a specific key.

This module contains two implementations:
1. AI-suggested approach (using sorted() with lambda)
2. Manual implementation (using list.sort() with operator.itemgetter)
"""

from operator import itemgetter


def sort_by_key_ai_suggested(data, key):
    """
    AI-suggested approach: Using sorted() with lambda function.
    
    Args:
        data: List of dictionaries to sort
        key: Key to sort by
        
    Returns:
        Sorted list of dictionaries
    """
    return sorted(data, key=lambda x: x[key])


def sort_by_key_manual(data, key):
    """
    Manual implementation: Using list.sort() with operator.itemgetter.
    
    Args:
        data: List of dictionaries to sort
        key: Key to sort by
        
    Returns:
        Sorted list of dictionaries (modifies original list in-place)
    """
    data.sort(key=itemgetter(key))
    return data


def sort_by_key_manual_copy(data, key):
    """
    Manual implementation with copy: Using list.sort() on a copy.
    Preserves original list.
    
    Args:
        data: List of dictionaries to sort
        key: Key to sort by
        
    Returns:
        Sorted list of dictionaries (original list unchanged)
    """
    result = data.copy()
    result.sort(key=itemgetter(key))
    return result


# Example usage and testing
if __name__ == "__main__":
    # Sample data
    sample_data = [
        {"name": "Alice", "age": 30, "score": 85},
        {"name": "Bob", "age": 25, "score": 92},
        {"name": "Charlie", "age": 35, "score": 78},
        {"name": "Diana", "age": 28, "score": 95},
    ]
    
    print("Original data:")
    for item in sample_data:
        print(item)
    
    print("\n--- AI-suggested approach (sorted by age) ---")
    sorted_ai = sort_by_key_ai_suggested(sample_data.copy(), "age")
    for item in sorted_ai:
        print(item)
    
    print("\n--- Manual implementation (sorted by score) ---")
    sorted_manual = sort_by_key_manual_copy(sample_data.copy(), "score")
    for item in sorted_manual:
        print(item)
    
    print("\n--- Testing with different keys ---")
    test_data = sample_data.copy()
    sorted_by_name = sort_by_key_ai_suggested(test_data, "name")
    print("Sorted by name:")
    for item in sorted_by_name:
        print(item)

