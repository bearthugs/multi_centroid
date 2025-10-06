import numpy as np
import os
from sklearn.decomposition import TruncatedSVD

def read_fvecs(filename):
    """Reads a .fvecs file and returns a NumPy array of shape (n, d)."""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    d = data.view(np.int32)[0]  # first int is the dimension
    return data.reshape(-1, d + 1)[:, 1:]  # drop the first value of each vector

def save_fvecs(filename, vectors):
    """Saves a NumPy array to .fvecs format."""
    n, d = vectors.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            np.array([d], dtype=np.int32).tofile(f)
            vectors[i].astype(np.float32).tofile(f)

def reduce_dimensionality(input_path, output_folder, target_dim=100):

    # Load vectors
    print(f"Loading vectors from {input_path}...")
    vectors = read_fvecs(input_path)

    if target_dim >= vectors.shape[1]:
        print(f"Original dimensionality ({vectors.shape[1]}) <= target ({target_dim}). Skipping reduction.")
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_file = os.path.join(output_folder, f"{base_name}_reduced.fvecs")

        save_fvecs(output_file, vectors)
        print(f"Saved unchanged vectors to: {output_file}")
        return

    print(f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")

    # Limit target dimension
    target_dim = min(target_dim, vectors.shape[1])

    # Apply Truncated SVD
    print(f"Reducing to {target_dim} dimensions using SVD...")
    svd = TruncatedSVD(n_components=target_dim, random_state=42)
    reduced_vectors = svd.fit_transform(vectors)
    print("Dimensionality reduction complete!")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save result
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}_reduced.fvecs")
    save_fvecs(output_file, reduced_vectors)
    print(f"Saved reduced vectors to: {output_file}")
    print("Original:", vectors.shape)
    print("Reduced:", reduced_vectors.shape)
    print("Explained variance ratio sum:", svd.explained_variance_ratio_.sum())

# Example usage
if __name__ == "__main__":
    train_files = ["fvecs_data/coco-i2i-512-angular_train.fvecs", "fvecs_data/deep-image-96-angular_train.fvecs",
                   "fvecs_data/fashion-mnist-784-euclidean_train.fvecs", "fvecs_data/gist-960-euclidean_train.fvecs",
                   "fvecs_data/glove-25-angular_train.fvecs", "fvecs_data/glove-50-angular_train.fvecs",
                   "fvecs_data/glove-100-angular_train.fvecs", "fvecs_data/glove-200-angular_train.fvecs",
                   "fvecs_data/lastfm-64-dot_train.fvecs", "fvecs_data/mnist-784-euclidean_train.fvecs",
                   "fvecs_data/nytimes-256-angular_train.fvecs", "fvecs_data/sift-128-euclidean_train.fvecs"]
    
    for train_file in train_files:
        input_file = train_file
        output_dir = "reduced_data"
        reduce_dimensionality(input_file, output_dir, target_dim=100)

