import sys, joblib, numpy as np
from sklearn.preprocessing import normalize

svd_path, query_path = sys.argv[1], sys.argv[2]

# load query vector
query = np.loadtxt(query_path, dtype=np.float32).reshape(1, -1)
svd = joblib.load(svd_path)
query_reduced = svd.transform(query)
np.savetxt("query_reduced.txt", query_reduced)
