import os
import zipfile
import requests
from io import BytesIO
import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist

GITHUB_API_URL = "https://api.github.com/repos"
GITHUB_REPO = "nick-oskiper/Thingiverse-Scraper"
ACCESS_TOKEN = os.getenv("GITHUB_TOKEN")
DOWNLOAD_PATH = r"C:\Users\noski\download_files\downloaded_files"
MAX_FILES = 30

def download_and_extract_files(repo, path, token, download_path, max_files):
    headers = {"Authorization": f"token {token}"}
    url = f"{GITHUB_API_URL}/{repo}/contents/{path}"
    response = requests.get(url, headers=headers)
    file_count = 0
    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            if file_count >= max_files:
                break
            if item["type"] == "dir":
                sub_files_count = download_and_extract_files(repo, item["path"], token, download_path, max_files - file_count)
                file_count += sub_files_count
            elif item["name"].endswith(".zip"):
                download_url = item["download_url"]
                zip_response = requests.get(download_url, headers=headers)
                extract_path = os.path.join(download_path, os.path.splitext(item["name"])[0])
                with zipfile.ZipFile(BytesIO(zip_response.content)) as zip_ref:
                    zip_ref.extractall(extract_path)
                file_count += 1
                print(f"Extracted {item['name']} to {extract_path}")
    else:
        print(f"Failed to retrieve contents from {url}")
    return file_count

def load_3d_model(file_path):
    try:
        print(f"Attempting to load 3D model from: {file_path}")
        return trimesh.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def convert_to_point_cloud(mesh):
    vertices = np.asarray(mesh.vertices)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    return point_cloud

def extract_features(point_cloud, max_points=1000):
    if len(point_cloud.points) > max_points:
        point_cloud = point_cloud.uniform_down_sample(every_k_points=len(point_cloud.points) // max_points)
    points = np.asarray(point_cloud.points)
    return points

def chamfer_distance(points1, points2):
    dist1 = cdist(points1, points2).min(axis=1).mean()
    dist2 = cdist(points2, points1).min(axis=1).mean()
    return dist1 + dist2

def find_similar_models(base_path, target_id, threshold=10000, max_points=1000):
    target_features = None
    target_model_path = None

    # Extract the target model features
    for root, dirs, files in os.walk(base_path):
        if str(target_id) in os.path.basename(root):
            print(f"Found target directory: {root}")
            files_path = os.path.join(root, "files")
            print(f"Looking for model files in: {files_path}")
            if os.path.exists(files_path):
                print(f"Files path exists: {files_path}")
                for file in os.listdir(files_path):
                    if file.endswith(('.stl', '.obj')):
                        target_model_path = os.path.join(files_path, file)
                        print(f"Found target file: {target_model_path}")
                        target_model = load_3d_model(target_model_path)
                        if target_model:
                            point_cloud = convert_to_point_cloud(target_model)
                            target_features = extract_features(point_cloud, max_points)
                            print(f"Extracted features from target model")
                            break
                        else:
                            print(f"Failed to load model from file: {target_model_path}")
            else:
                print(f"Files path does not exist: {files_path}")
            break
        else:
            print(f"Skipping directory: {root}")

    if target_features is None:
        raise ValueError(f"Target model {target_id} not found or could not be loaded.")

    similar_ids = []

    # Compare with all other models
    for root, dirs, files in os.walk(base_path):
        if str(target_id) not in os.path.basename(root):
            files_path = os.path.join(root, "files")
            if os.path.exists(files_path):
                for file in os.listdir(files_path):
                    if file.endswith(('.stl', '.obj')):
                        model_path = os.path.join(files_path, file)
                        model = load_3d_model(model_path)
                        if model:
                            point_cloud = convert_to_point_cloud(model)
                            features = extract_features(point_cloud, max_points)
                            distance = chamfer_distance(target_features, features)
                            print(f"Comparing with {model_path}, distance: {distance}")
                            if distance < threshold:
                                similar_ids.append(os.path.basename(root))
                                break

    return similar_ids

# Main script execution
if __name__ == "__main__":
    base_path = DOWNLOAD_PATH
    target_id = 2555039  # Example target ID

    # Download and extract files
    print(f"Starting to download and extract files to {base_path}")
    download_and_extract_files(GITHUB_REPO, "downloaded_files", ACCESS_TOKEN, base_path, MAX_FILES)

    # Find similar models
    print(f"Starting to find similar models for target ID {target_id}")
    similar_models = find_similar_models(base_path, target_id)
    print(f"Models similar to {target_id}: {similar_models}")
