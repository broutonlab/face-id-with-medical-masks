import os
import cv2
import numpy as np
import base64
import random
from tqdm import tqdm
import face_alignment

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')


def image_to_string(image: np.ndarray) -> str:
    return base64.b64encode(
        cv2.imencode('.png', image)[1]
    ).decode()


def string_to_image(s: str) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(base64.b64decode(s), dtype=np.uint8),
        flags=1
    )


def warp_affine_to_points(
        points: np.ndarray,
        affine_transform_matrix: np.ndarray) -> np.ndarray:
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    transformed_points = affine_transform_matrix.dot(points_ones.T).T

    return transformed_points.astype(np.int32)


def rotate_image_and_points(
        mat: np.ndarray, angle: float, points: np.ndarray) -> tuple:
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (width / 2,
                    height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    rotated_points = warp_affine_to_points(points, rotation_mat)

    return rotated_mat, rotated_points


def draw_landmarks(
        image: np.ndarray,
        points: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 3) -> np.ndarray:
    result = image.copy()
    for p in points:
        result = cv2.circle(result, tuple(p), thickness, color, thickness // 2,
                            -1)
    return result


def l2_measure(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(((a - b) ** 2).sum())


def extract_mask_points(points: np.ndarray) -> np.ndarray:
    target_mask_polygon_points = np.zeros((16, 2), dtype=np.int32)

    target_mask_polygon_points[0] = points[28].astype(np.int32)
    target_mask_polygon_points[1:] = points[1:16].astype(np.int32)

    return target_mask_polygon_points


def extract_target_points_and_characteristic(points: np.ndarray) -> tuple:
    avg_left_eye_point = points[36:42].mean(axis=0)
    avg_right_eye_point = points[42:48].mean(axis=0)
    avg_mouth_point = points[48:68].mean(axis=0)

    left_face_point = points[1]
    right_face_point = points[15]

    d1 = l2_measure(left_face_point, avg_mouth_point)
    d2 = l2_measure(right_face_point, avg_mouth_point)

    x1, y1 = avg_left_eye_point
    x2, y2 = avg_right_eye_point
    alpha = np.arctan((y2 - y1) / (x2 - x1 + 1E-5))

    s1 = alpha * 180 / np.pi
    s2 = d1 / (d2 + 1E-5)

    target_mask_polygon_points = extract_mask_points(points)

    return target_mask_polygon_points, s1, s2


def extract_polygon(image: np.ndarray, points: np.ndarray) -> tuple:
    rect = cv2.boundingRect(points)
    x1, y1, w, h = rect
    x2, y2 = x1 + w, y1 + h

    crop = image[y1:y2, x1:x2]
    shifted_points = points - np.array([x1, y1], dtype=np.int32)

    crop_mask = cv2.fillPoly(
        np.zeros((h, w), dtype=np.uint8),
        [shifted_points],
        (255)
    )

    crop[crop_mask == 0] = 0

    rgba_crop = np.concatenate(
        (
            crop,
            np.expand_dims(crop_mask, axis=2)
        ),
        axis=2
    )

    return rgba_crop, shifted_points


def end2end_mask_encoding(
        image: np.ndarray,
        face_aligment_class: face_alignment.FaceAlignment) -> dict:
    landmarks = face_aligment_class.get_landmarks_from_image(image)
    landmarks = np.floor(landmarks[0]).astype(np.int32)

    target_points, s1, s2 = extract_target_points_and_characteristic(landmarks)
    mask_rgba_crop, target_points = extract_polygon(image,
                                                    target_points)

    mask_rgba_crop, target_points = rotate_image_and_points(mask_rgba_crop, s1,
                                                            target_points)

    res = {
        's1': s1,
        's2': s2,
        'points': target_points.tolist(),
        'base64_img': image_to_string(mask_rgba_crop)
    }

    return res


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def get_traingulation_mesh_points_indexes(points):
    rect = cv2.boundingRect(points)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points.tolist())

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return np.array(indexes_triangles, dtype=np.int32)


def get_transforms_list_between_point_triangles(
        traingles_indexing_map: np.ndarray,
        points1: np.ndarray,
        points2: np.ndarray) -> np.ndarray:
    assert traingles_indexing_map.shape[1] == 3

    triangles_transform_matrixes = np.zeros(
        (len(traingles_indexing_map), 2, 3),
        dtype=np.float32
    )

    for i in range(len(traingles_indexing_map)):
        triangles_transform_matrixes[i] = cv2.getAffineTransform(
            points1[traingles_indexing_map[i]].astype(np.float32),
            points2[traingles_indexing_map[i]].astype(np.float32)
        )

    return triangles_transform_matrixes


def move_triangle_to_image(
        src_image: np.ndarray,
        dst_image: np.ndarray,
        triangle: np.ndarray,
        affine_transform_matrix: np.ndarray) -> np.ndarray:
    triangle_mask = cv2.fillConvexPoly(
        np.zeros(
            (src_image.shape[0], src_image.shape[1]),
            dtype=np.uint8
        ),
        triangle,
        (255)
    )

    warped_triangle_area = cv2.warpAffine(
        src_image,
        affine_transform_matrix,
        (dst_image.shape[1], dst_image.shape[0])
    )

    warped_triangle_mask = cv2.warpAffine(
        triangle_mask,
        affine_transform_matrix,
        (dst_image.shape[1], dst_image.shape[0])
    )

    gray_waped_triangle_area = cv2.cvtColor(warped_triangle_area,
                                            cv2.COLOR_RGB2GRAY)
    _, mask_gray_waped_triangle_area = cv2.threshold(
        gray_waped_triangle_area,
        20,
        255,
        cv2.THRESH_BINARY
    )

    warped_triangle_mask[gray_waped_triangle_area < 50] = 0

    dst_image[warped_triangle_mask > 0] = warped_triangle_area[
        warped_triangle_mask > 0]

    return dst_image


def warp_mask(
        mask_image: np.ndarray,
        dst_image: np.ndarray,
        mask_landmarks: np.ndarray,
        dst_landmarks: np.ndarray) -> np.ndarray:
    indexes_map = get_traingulation_mesh_points_indexes(dst_landmarks)
    affs = get_transforms_list_between_point_triangles(indexes_map,
                                                       mask_landmarks,
                                                       dst_landmarks)

    mask_warp = dst_image.copy()
    for i in range(len(indexes_map)):
        mask_warp = move_triangle_to_image(
            mask_image,
            mask_warp,
            mask_landmarks[indexes_map[i]],
            affs[i]
        )

    mask_area = cv2.fillConvexPoly(
        np.zeros((dst_image.shape[0], dst_image.shape[1]), dtype=np.uint8),
        dst_landmarks,
        (255)
    )

    result = dst_image.copy()
    result[mask_area > 0] = mask_warp[mask_area > 0]

    return result


def apply_mask_to_image_with_face(
        image: np.ndarray,
        mask_data: dict,
        face_landmarks: np.ndarray) -> np.ndarray:

    mask_image = string_to_image(mask_data['base64_img'])

    target_points, _, _ = extract_target_points_and_characteristic(
        face_landmarks
    )

    face_image_with_mask = warp_mask(
        mask_image[..., :3],
        image,
        np.array(mask_data['points'], dtype=np.int32),
        target_points
    )

    return face_image_with_mask


def extract_mask_from_base(
        masks_base: dict,
        d: float,
        ci: float = 0.01):
    """
    Extract random mask from nearest masks by parameter d
    Args:
        masks_base: dictionary with masks images data
        d: target parameter
        ci: confidence interval for sampling

    Returns:
        Sampling element
    """

    diff_rates = [
        (i, abs(e['s2'] - d))
        for i, e in enumerate(masks_base['masks'])
    ]

    diff_rates.sort(key=lambda x: x[1])

    filtered_indexes_by_ci_level = [
        dr[0]
        for dr in diff_rates
        if dr[1] - ci < 1E-5
    ]

    if len(filtered_indexes_by_ci_level) > 0:
        return masks_base['masks'][
            random.sample(filtered_indexes_by_ci_level, 1)[0]
        ]

    return masks_base['masks'][diff_rates[0][0]]


def end2end_mask_generation(
        image: np.ndarray,
        masks_database: dict,
        face_aligment_class: face_alignment.FaceAlignment,
        input_face_landmarks: np.ndarray = None):

    if input_face_landmarks is None:
        face_landmarks = face_aligment_class.get_landmarks_from_image(image)

        if len(face_landmarks) == 0:
            raise RuntimeError('Can\'t find facial landmarks')

        face_landmarks = np.floor(face_landmarks[0]).astype(np.int32)
    else:
        face_landmarks = input_face_landmarks

    _, _, s2 = extract_target_points_and_characteristic(
        face_landmarks
    )

    sampling_mask_data = extract_mask_from_base(
        masks_database,
        s2
    )

    face_with_mask = apply_mask_to_image_with_face(
        image,
        sampling_mask_data,
        face_landmarks
    )

    return face_with_mask


def generate_masks_base(
        masks_folder: str,
        face_aligment_class: face_alignment.FaceAlignment,
        verbose: bool = True,
        skip_warnings: bool = False) -> dict:
    masked_faces_images_pathes = [
        os.path.join(masks_folder, img_name)
        for img_name in os.listdir(masks_folder)
    ]

    result = {
        'masks': []
    }

    loop_generator = tqdm(masked_faces_images_pathes) \
        if verbose else masked_faces_images_pathes

    for img_path in loop_generator:
        image = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR
        )

        if image is None:
            print('Warning! Can\'t open follow image: {}'.format(img_path))
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        try:
            result['masks'].append(
                end2end_mask_encoding(image, face_aligment_class)
            )
        except Exception as e:
            if not skip_warnings:
                print('Failed prepare image {} because: {}'.format(img_path, e))

    return result
