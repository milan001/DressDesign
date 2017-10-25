import prettytensor as pt
import tensorflow as tf
import pickle


with open('../Category\ and\ Attribute\ Prediction\ Benchmark/Anno/list_attr_img.txt', 'rb') as f:
	embeddings = f.read(f)