import numpy as np
import os
import joblib
from sklearn.decomposition import TruncatedSVD


def read_fvecs(filename):
    """Reads a .fvecs file and returns a NumPy array of shape (n, d)."""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    if len(data) == 0:
        raise ValueError(f"{filename} is empty or invalid.")
    d = data.view(np.int32)[0]
    return data.reshape(-1, d + 1)[:, 1:]


def save_fvecs(filename, vectors):
    """Saves a NumPy array to .fvecs format."""
    n, d = vectors.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            np.array([d], dtype=np.int32).tofile(f)
            vectors[i].astype(np.float32).tofile(f)


def apply_or_copy(test_path, model_path, output_folder):
    base_name = os.path.splitext(os.path.basename(test_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}_reduced.fvecs")

    print(f"\n=== Processing {base_name} ===")
    print(f"Loading test vectors from: {test_path}")
    test_vectors = read_fvecs(test_path)
    print(f"Loaded {test_vectors.shape[0]} vectors of dim {test_vectors.shape[1]}")

    if os.path.exists(model_path):
        print(f"Applying existing SVD model: {model_path}")
        svd = joblib.load(model_path)
        reduced_vectors = svd.transform(test_vectors)
        print("Reduction complete.")
    else:
        print(f"No SVD model found for {base_name}. Skipping reduction (saving original vectors).")
        reduced_vectors = test_vectors

    os.makedirs(output_folder, exist_ok=True)
    save_fvecs(output_file, reduced_vectors)

    print(f"Saved reduced test vectors to: {output_file}")
    print("Original:", test_vectors.shape)
    print("Reduced :", reduced_vectors.shape)


if __name__ == "__main__":
    test_files = [
        "../fvecs_data/coco-i2i-512-angular_test.fvecs",
        "../fvecs_data/fashion-mnist-784-euclidean_test.fvecs",
        "../fvecs_data/gist-960-euclidean_test.fvecs",
        "../fvecs_data/glove-25-angular_test.fvecs",
        "../fvecs_data/glove-50-angular_test.fvecs",
        "../fvecs_data/glove-100-angular_test.fvecs",
        "../fvecs_data/glove-200-angular_test.fvecs",
        "../fvecs_data/lastfm-64-dot_test.fvecs",
        "../fvecs_data/mnist-784-euclidean_test.fvecs",
        "../fvecs_data/nytimes-256-angular_test.fvecs",
        "../fvecs_data/sift-128-euclidean_test.fvecs",
    ]

    model_dir = "../reduced_data/train"
    output_dir = "../reduced_data/test"

    for test_file in test_files:
        base_name = os.path.splitext(os.path.basename(test_file))[0].replace("_test", "_train")
        model_file = os.path.join(model_dir, f"{base_name}_svd_model.pkl")

        apply_or_copy(test_file, model_file, output_dir)
