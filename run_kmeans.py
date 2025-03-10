import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import argparse
import warnings

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="Full path of the dataset", type=str)
    parser.add_argument("k", help="Number of clusters", type=int)
    parser.add_argument("n_init", help="Number of times the k-means algorithm is run with different centroid seeds", type=int)
    parser.add_argument("assignment_file", help="Full path of the assignment vector", type=str)

    args = parser.parse_args()

    X = pd.read_csv(args.data_file, sep=" ", header=None)
    
    # print('Input data shape:', X.shape)
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X_norm = scaler.transform(X)


    kmeans = KMeans(n_clusters=args.k, n_init=args.n_init).fit(X)
    data = [kmeans.inertia_] + list(kmeans.labels_)
    pd.DataFrame(data).to_csv(args.assignment_file, sep=" ", index=False, header=False)


