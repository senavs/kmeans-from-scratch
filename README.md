# kmeans-from-scratch
A  Python implementation of KMeans machine learning algorithm.

## Algorithm
[K-means clustering](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1) is one of the simplest and popular unsupervised machine learning algorithms. It's identifies _k_ number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.

## Clusters
<p align="center">
  <img src="https://github.com/senavs/kmeans-from-scratch/blob/master/notebook/kmeans-graph-readme.png" width=500>
</p>

## Implementation
[Point](https://github.com/senavs/kmeans-from-scratch/blob/master/model/point.py) is a class to represent a point in cartesian plane. You are able to sum, subtract, multiply, divide and calculate distance between two points. You can read more about Point class in my [knn-from-scratch repository](https://github.com/senavs/knn-from-scratch) where I demonstrated in more details.  
[KMeans](https://github.com/senavs/kmeans-from-scratch/blob/master/model/kmeans.py) is the model class. Only the methods are allowed: `fit` and `predict`. Look into `help(KMeans)` for more infomraiton.
```python
from model.kmeans import KMeans

kmeans = KMeans(k=5, seed=101)
kmeans.fit(x_train, epochs=100)

predict = kmeans.predict(x_predict)
```

## Apply KMeans from scratch in dataset
To show the package working, I created a jupyter notebook. Take a look into [here](https://github.com/senavs/kmeans-from-scratch/blob/master/notebook/kmeans-random_dataset.ipynb).
