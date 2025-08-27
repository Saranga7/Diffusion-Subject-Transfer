import kagglehub

# Download latest version
path = kagglehub.dataset_download("zeegelin/cats-and-dogs-small")

print("Path to dataset files:", path)