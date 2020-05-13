root_dataset_path = '/media/alexey/DataDisk/datasets/vggface2_with_face_boxes_markup/test/'

from masked_face_sdk.pipeline_dataset_loader \
    import PipelineFacesDatasetGenerator
from masked_face_sdk.neural_network_modules \
    import Backbone, ArcFaceLayer, FaceRecognitionModel, resnet18

from torch.utils.data import DataLoader

batch_size = 32
n_jobs = 4
epochs = 20
image_shape = (112, 112)
embedding_size = 128

generator_train_dataset = PipelineFacesDatasetGenerator(
    root_dataset_path,
    image_shape
)

train_data = DataLoader(
        generator_train_dataset,
        batch_size=batch_size,
        num_workers=n_jobs,
        shuffle=True,
        drop_last=True
)

model = FaceRecognitionModel(
    backbone=Backbone(
        backbone=resnet18(pretrained=True),
        embedding_size=embedding_size,
        input_shape=(3, image_shape[0], image_shape[1])
    ),
    head=ArcFaceLayer(
        embedding_size=embedding_size,
        num_classes=generator_train_dataset.num_classes
    )
)

loss = torch.nn.CrossEntropyLoss()



