from argparse import ArgumentParser
from masked_face_sdk.mask_generation_utils import end2end_mask_generation
from masked_face_sdk.crop_utils import create_square_crop_by_detection
import face_alignment
import json
import os
import numpy as np
import cv2
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description='Apply masks transform to face dataset')

    parser.add_argument(
        '--face-dataset-folder', required=True, type=str,
        help='Path to folder with dataset in Keras format.'
    )

    parser.add_argument(
        '--masks-database-file', required=True, type=str,
        help='Path to json masks database file.'
    )

    parser.add_argument(
        '--verbose', action='store_true'
    )

    parser.add_argument(
        '--skip-warnings', action='store_true'
    )

    parser.add_argument(
        '--use-cuda', action='store_true'
    )

    return parser.parse_args()


def get_box_by_facial_landmarks(
        landmarks: np.ndarray,
        additive_coefficient: float = 0.2) -> list:
    """
    Configure face bounding box by landmarks points
    Args:
        landmarks: landmarks points in int32 datatype and XY format
        additive_coefficient: value of additive face area on box

    Returns:
        List with bounding box data in XYXY format
    """
    x0 = landmarks[..., 0].min()
    y0 = landmarks[..., 1].min()

    x1 = landmarks[..., 0].max()
    y1 = landmarks[..., 1].max()

    dx = int((x1 - x0) * additive_coefficient)
    dy = int((y1 - y0) * additive_coefficient * 2)

    return [
        x0 - dx // 2, y0 - dx // 2,
        x1 + dx // 2, y1 + dy // 2
    ]


def main():
    args = parse_args()

    with open(args.masks_database_file, 'r') as jf:
        masks_database = json.load(jf)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device='cpu' if not args.use_cuda else 'cuda'
    )

    samples_pathes = sorted(
        [
            os.path.join(args.face_dataset_folder, sname)
            for sname in os.listdir(args.face_dataset_folder)
        ]
    )

    loop_generator = tqdm(samples_pathes) if args.verbose else samples_pathes

    for face_folder in loop_generator:
        for image_name in os.listdir(face_folder):
            image_path = os.path.join(face_folder, image_name)

            try:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    if not args.skip_warnings:
                        print(
                            'Skipping and delete from base follow image: '
                            '{}, because can\'t open this image'.format(
                                image_path
                            )
                        )
                    os.remove(image_path)
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                face_landmarks = fa.get_landmarks_from_image(image)

                if face_landmarks is None or len(face_landmarks) == 0:
                    if not args.skip_warnings:
                        print(
                            'Skipping and delete from base follow image: '
                            '{}, because can\'t find faces landmarks'.format(
                                image_path
                            )
                        )
                    os.remove(image_path)
                    continue

                face_landmarks = np.floor(face_landmarks[0]).astype(np.int32)

                face_box = get_box_by_facial_landmarks(face_landmarks)

                face_with_mask = end2end_mask_generation(
                    image,
                    masks_database,
                    None,
                    face_landmarks
                )

                new_face_crop = create_square_crop_by_detection(
                    face_with_mask,
                    face_box
                )

                save_status = cv2.imwrite(
                    image_path,
                    cv2.cvtColor(new_face_crop, cv2.COLOR_RGB2BGR)
                )

                if not save_status:
                    if not args.skip_warnings:
                        print(
                            'Skipping and delete from base follow image: '
                            '{}, because can\'t save parsed this image'.format(
                                image_path
                            )
                        )
                    os.remove(image_path)
                    continue
            except Exception as e:
                if not args.skip_warnings:
                    print(
                        'Skipping and delete from base follow image: '
                        '{}, because unexpected error: {}'.format(
                            e, image_path
                        )
                )
                os.remove(image_path)


if __name__ == '__main__':
    main()
