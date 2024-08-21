import os
import re
import sys
import shutil
import pathlib
import pymeshlab
# import traceback
import threading
import numpy as np
import pandas as pd
from time import time
import scipy.sparse.linalg
# from helpers import timeout
from edge import edge_collapse
import matplotlib.pyplot as plt
from mesh_graph import MeshGraph
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import PCA
from pathlib import Path
import zipfile
from io import BytesIO
import requests

### Global Constants ###
GITHUB_API_URL = "https://api.github.com/repos"
GITHUB_REPO = "nick-oskiper/Thingiverse-Scraper"
ACCESS_TOKEN = os.getenv("GITHUB_TOKEN")
DOWNLOAD_PATH = r"C:\Users\noski\download_files\downloaded_files"
MAX_FILES = 1000

yes = {"yes", "yeah", "y", "yea"}
colors = [
    {"centroid": "red", "points": "red"},
    {"centroid": "green", "points": "green"},
    {"centroid": "black", "points": "black"},
    {"centroid": "yellow", "points": "yellow"},
    {"centroid": "blue", "points": "blue"},
    {"centroid": "orange", "points": "orange"},
    {"centroid": "purple", "points": "purple"},
    {"centroid": "pink", "points": "pink"},
    {"centroid": "violet", "points": "violet"},
    {"centroid": "brown", "points": "brown"},
    {"centroid": "gray", "points": "gray"},
    {"centroid": "red", "points": "blue"},
    {"centroid": "red", "points": "blue"},
    {"centroid": "red", "points": "blue"},
    {"centroid": "red", "points": "blue"},
    {"centroid": "red", "points": "blue"},
    {"centroid": "red", "points": "blue"},
    {"centroid": "red", "points": "blue"},
    {"centroid": "red", "points": "blue"},
    {"centroid": "red", "points": "blue"},
    {"centroid": "red", "points": "blue"},
]
results_dir = "/Users/anniepates/Documents/DREAM/fa3ds/backend/segmentation_experiment"
# default_models_dir = "/Users/anniepates/Documents/DREAM/fa3ds/backend/segmentation_experiment"
#report = lambda error: f"\033[31m----------------------------\n{error}\n----------------------------\033[0m\n"


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
                sub_files_count = download_and_extract_files(repo, item["path"], token, download_path,
                                                             max_files - file_count)
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


# @timeout(300)
def segment_mesh(mesh, k=None, collapsed=False):
    print(f"--- Segmenting Mesh ---\nfaces: {mesh.face_matrix().shape}\nvertices: {mesh.vertex_matrix().shape}")
    start_time = time()

    mesh_graph = MeshGraph(mesh, collapsed=collapsed)
    similarity_matrix = mesh_graph.similarity_matrix(collapsed=collapsed)
    sqrt_degree = np.sqrt(mesh_graph.collapsed_degree_matrix) if collapsed else np.sqrt(mesh_graph.degree_matrix)
    laplacian = sqrt_degree.T * similarity_matrix.T * sqrt_degree

    if k is None:
        eigen_values_full, _ = scipy.linalg.eigh(laplacian)
        eigen_values_full = np.sort(eigen_values_full)
        eigen_std = np.std(eigen_values_full)
        k = len([abs(eigen_values_full[i] - eigen_values_full[i - 1]) for i in range(1, len(eigen_values_full)) if abs(eigen_values_full[i] - eigen_values_full[i - 1]) >= eigen_std])
        print(f"\033[32mRecommended segment count: {k}\033[0m")

    print(f"Segmenting mesh into {k} segments")
    eigen_values, eigen_vectors = scipy.sparse.linalg.eigsh(laplacian)
    eigen_vectors /= np.linalg.norm(eigen_vectors, axis=1)[:, None]

    _, labels = kmeans2(eigen_vectors, k, minit="++", iter=50)
    visualize_eigen_vectors(eigen_vectors, k)

    if collapsed:
        labels = _unwrap_labels(mesh_graph, labels)

    print(f"\033[33mSegmented mesh into {len(set(labels))} segments\033[0m")
    print(f"--- Done ---")
    return k, labels


def extract_segments(vertices, faces, labels, num_segments, t, parent_dir=results_dir):
    segmentation_dir = parent_dir

    segment_indices = {}
    for j in range(len(labels)):
        label = labels[j]
        if label not in segment_indices:
            os.makedirs(f"{segmentation_dir}/segment_{label}", exist_ok=True)
            segment_indices[label] = []
        segment_indices[label].append(j)

    for label, indices in segment_indices.items():
        segment_faces = np.zeros((len(indices), 3))
        vertices_map = {}
        segment_vertices = []

        for j, k in enumerate(indices):
            segment_face = []
            for v in faces[k]:
                if v not in vertices_map:
                    vertices_map[v] = len(segment_vertices)
                    segment_vertices.append(vertices[v])
                segment_face.append(vertices_map[v])
            segment_faces[j] = np.array(segment_face)

        ms = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(np.array(segment_vertices), segment_faces)
        ms.add_mesh(mesh)
        ms.save_current_mesh(f"{segmentation_dir}/segment_{label}/segment_{label}.obj")

    df = pd.DataFrame([[segmentation_dir, faces.shape[0], num_segments, t]], columns=["meshPath", "resolution", "numSegments", "segmentationTime"])
    df.to_csv(f"{parent_dir}/info.csv", mode='a', encoding='utf-8', index=False)

def visualize_eigen_vectors(eigen_vectors, k, n=2, reduced=True):
    """
    Visualizes a set of eigen vectors (with reduced dimensionality to 2 or 3)
    """
    pca_reducer = PCA(n, svd_solver='full')
    reduced_eigen_vectors = pca_reducer.fit_transform(eigen_vectors)
    reduced_eigen_vectors_sep = [
        reduced_eigen_vectors.T[i, :]
        for i in range(n)
    ]
    centroids, labels = kmeans2(reduced_eigen_vectors, k, minit="++", iter=50)

    if n == 2:
        x, y = reduced_eigen_vectors_sep
        for i in range(k):
            plt.scatter(centroids[i][0], centroids[i][1], color=colors[i]['centroid'])

            plt.scatter(
                [x[j] for j in range(len(x)) if labels[j] == i],
                [y[j] for j in range(len(y)) if labels[j] == i],
                color=colors[i]['points']
            )
    else:
        print(f"Support not yet added for visualizing {n}-D reduced eigen vectors")
        return

    # if os.path.exists(f"{results_dir}/eigenvectors.png"): os.remove(f"{results_dir}/eigenvectors.png")
    plt.savefig(f"{results_dir}/eigenvectors.png")


### Helper Classes ###
class thread(threading.Thread):
    def __init__(self, thread_id, f, args):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.args = args
        self.f = f

        # helper function to execute the threads

    def run(self):
        print(f"[{self.thread_id}] Starting ...")
        return self.f(self.args)


### Helper Functions ####
def _predict_segmentation(mesh, face_count, collapsed=False):
    """
    """
    mesh = edge_collapse(mesh, face_count=face_count)
    # Step 1
    mesh_graph = MeshGraph(mesh, collapsed)

    # Step 2
    similarity_matrix = mesh_graph.similarity_matrix(collapsed=collapsed)

    # Step 3
    sqrt_degree = np.sqrt(mesh_graph.collapsed_degree_matrix) if collapsed else np.sqrt(mesh_graph.degree_matrix)

    laplacian = sqrt_degree.T * similarity_matrix.T * sqrt_degree
    eigen_values_full, _ = scipy.linalg.eigh(laplacian)
    eigen_values_full = np.sort(eigen_values_full)
    eigen_std = np.std(eigen_values_full)
    num_segments = len(
        [abs(eigen_values_full[i] - eigen_values_full[i - 1]) for i in range(1, len(eigen_values_full)) if
         abs(eigen_values_full[i] - eigen_values_full[i - 1]) >= eigen_std])

    print(f"\033[32mRecommended segment count for {face_count} resolution: {num_segments}\033[0m")

    return mesh, num_segments


def _remesh(mesh, save_path=None):
    """
    Resmeshes a mesh to force all faces to become polygons

    Inputs
        :mesh: <pymeshlab.Mesh> mesh to be remeshed

    Throws
        <ValueError> if the number of verticies > number of vertex normals
    """
    ms = pymeshlab.MeshSet()
    if isinstance(mesh, str):
        ms.load_new_mesh(mesh)
    else:
        ms.add_mesh(mesh)
    # ms.meshing_isotropic_explicit_remeshing(iterations=3)
    # ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pymeshlab.Percentage(1.5))

    ms.compute_matrix_from_scaling_or_normalization(scalecenter='origin', unitflag=True, uniformflag=True)
    targetlen_given = 1.5

    attempts = 10
    while attempts > 0:
        ms.meshing_isotropic_explicit_remeshing(iterations=4, targetlen=pymeshlab.Percentage(targetlen_given))
        if (ms.current_mesh().face_matrix().shape[0] < 24000):
            print(f"[remesh] >> Optimizing resolution {ms.current_mesh().face_matrix().shape[0]}...")
            targetlen_given -= 0.1
        elif (ms.current_mesh().face_matrix().shape[0] > 26000):
            print(f"[remesh] >> Optimizing resolution {ms.current_mesh().face_matrix().shape[0]}...")
            targetlen_given += 0.1
        else:
            print(
                f"[remesh] >> Optimization completed. Current resolution: {ms.current_mesh().face_matrix().shape[0]}... ")
            break
        attempts -= 1

    for i in ms: pass
    if save_path is not None: ms.save_current_mesh(save_path)
    return ms.current_mesh()


def _construct_dir(dir_name, overwrite="yes"):
    """
    Constructs a new directory with name dir_name, if one does not exist
    else it promopts user to over-write

    Inputs
        :dir_name:
    """
    if os.path.exists(dir_name):
        if overwrite is None: overwrite = input(f"Directory {dir_name} exists, do you wish to overwrite [y, (n)]? ")
        if overwrite.lower() in yes:
            shutil.rmtree(dir_name)
            os.mkdir(dir_name)
    else:
        os.mkdir(dir_name)


def _remove_dir(dir_name):
    """
    Removes a directory and its contained items

    Inputs
        :dir_name:
    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def _unwrap_labels(mesh_graph, labels):
    """
    Unwraps the collapsed labels back to cover the entire mesh

    Inputs
        mesh_graph: <MeshGraph> a graph where every node is a set of faces and edges exist between adjacent set of faces
        labels: <np.ndarray> where the ith element is the label corresponding to the ith face set

    Outputs
        :returns: <list> where the ith element is the label corresponding to the ith face
    """
    n = len(mesh_graph.collapsed_map)

    unwrapped_labels = np.zeros(n)
    reverse_map = {}
    for i, j in mesh_graph.collapsed_map.items():
        try:
            reverse_map[j[1]].append(i)
        except:
            reverse_map[j[1]] = [i]

    for j, faces in reverse_map.items():
        for i in faces: unwrapped_labels[i] = labels[j]
    return unwrapped_labels

def process_meshes(n=None):
    download_and_extract_files(GITHUB_REPO, "downloaded_files", ACCESS_TOKEN, DOWNLOAD_PATH, MAX_FILES)

    download_dir = Path(DOWNLOAD_PATH)
    zip_files = list(download_dir.glob('**/*.zip'))

    if n is not None:
        zip_files = zip_files[:n]

    for zip_file in zip_files:
        model_id = zip_file.parent.name
        print(f"Processing model ID: {model_id}")
        extract_dir = zip_file.parent / 'extracted'
        # No need to extract again here, it's done in the previous step
        mesh_files = list(extract_dir.glob('**/*.stl')) + list(extract_dir.glob('**/*.obj'))
        for mesh_file in mesh_files:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(mesh_file))
            mesh = ms.current_mesh()

            start_time = time()

            k, labels = segment_mesh(mesh)
            extract_segments(mesh.vertex_matrix(), mesh.face_matrix(), labels, k, time() - start_time, parent_dir=f'segmentation_results/{model_id}')
            print(f"Finished processing {mesh_file.name}")



if __name__ == "__main__":
    process_meshes(n=5)  # Change n to number
