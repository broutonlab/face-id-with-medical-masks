import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
from scipy.spatial import distance_matrix


def default_acc_function(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).sum().type(
        torch.FloatTensor
    ) / y_true.size(0)


def l2(a: np.ndarray, b: np.ndarray):
    return np.sqrt(((a - b) ** 2).sum())


def test_embedding_net(
        faces_root_dir: str,
        image_shape: tuple,
        model: torch.nn.Module,
        device: str,
        verbose: bool = True):
    assert os.path.isdir(faces_root_dir)

    model.eval()

    faces_folders = [
        os.path.join(faces_root_dir, ff)
        for ff in os.listdir(faces_root_dir)
    ]

    persons_embeddings = []

    if verbose:
        print('Loading and process test dataset:')
    loop_generator = tqdm(faces_folders) if verbose else faces_folders

    for face_folder in loop_generator:
        person_embeddings = []
        for image_name in os.listdir(face_folder):
            image_path = os.path.join(face_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                if verbose:
                    print('Can\'t open image: {}'.format(image_path))
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(
                image,
                image_shape[::-1],
                interpolation=cv2.INTER_NEAREST
            )

            input_tensor = (
                                   torch.FloatTensor(
                                       image
                                   ).permute(2, 0, 1).unsqueeze(0) / 255.0 - 0.5
                           ) * 2
            input_tensor = input_tensor.to(device)

            output_embedding = model.inference(input_tensor).to('cpu').detach()
            embedding = output_embedding.numpy()[0]
            del input_tensor

            person_embeddings.append(embedding)

        persons_embeddings.append(person_embeddings)

    flatten_persons_embeddings = [
        (person_idx, emb)
        for person_idx, series in enumerate(persons_embeddings)
        for emb in series
    ]

    flatten_embeddings = np.array(
        [fpe[1] for fpe in flatten_persons_embeddings]
    )

    flatten_indexes = np.array(
        [fpe[0] for fpe in flatten_persons_embeddings]
    )

    print(flatten_embeddings.shape)

    if verbose:
        print('Evaluate accuracy rate...')
    pairwise_distances = distance_matrix(flatten_embeddings, flatten_embeddings)
    pairwise_distances[np.diag_indices_from(pairwise_distances)] = 1000.0

    mins = pairwise_distances.argmin(axis=1)
    return (flatten_indexes[mins] == flatten_indexes).sum() / len(
        flatten_persons_embeddings
    )
