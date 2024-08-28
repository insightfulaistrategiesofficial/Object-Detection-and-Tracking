from deep_sort.deep_sort import DeepSort
from config.config import deep_sort_weights, max_age

deep_sort_weights = deep_sort_weights # Path to the DeepSort model
tracker = DeepSort(model_path=deep_sort_weights, max_age=max_age) # Initialize the DeepSort tracker