import argparse
import numpy as np
from sklearn.cluster import KMeans

if '__main__' == __name__:
    p = argparse.ArgumentParser()
    p.add_argument("data_file", type=str)
    p.add_argument("k", type=int)
    p.add_argument("assignment_file", type=str)
    p.add_argument("only_value", type=int)

    args = p.parse_args()

    X = np.loadtxt(args.data_file, dtype=np.float32)
    km = KMeans(n_clusters=args.k, n_init=1000, copy_x=False, verbose=0).fit(X)
    
    if args.only_value == 1:
        with open(args.assignment_file, "w") as f:
            f.write(f"{km.inertia_:.17g}\n")
    else:
        np.savetxt(args.assignment_file, km.labels_.astype(np.int32), fmt="%d")


