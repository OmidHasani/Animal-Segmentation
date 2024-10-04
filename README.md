This project implements an object detection model using the RetinaNet architecture, which is built on top of ResNet50 as a backbone for feature extraction. The process involves:

1. **Data Preparation**: The dataset is downloaded, extracted, and preprocessed by resizing images and converting bounding boxes into the appropriate format.
  
2. **Anchor Box Generation**: Anchor boxes are generated for various aspect ratios and scales, which serve as reference points for detecting objects in the images.

3. **Model Architecture**: The model consists of ResNet50 to extract features from images and a Feature Pyramid Network (FPN) to generate multi-scale feature maps. The final class predictions and bounding box coordinates are made using separate heads (classification and box regression).

4. **Training**: The model is trained using a combination of Focal Loss for class imbalance and Smooth L1 Loss for bounding box regression. The training data is augmented and shuffled to improve the model's robustness.

5. **Inference**: After training, the model is used to predict objects in test images, visualizing the detected bounding boxes on the images.

In summary, this project demonstrates the implementation of a state-of-the-art object detection system capable of identifying and localizing objects in images.
---
Process of this project : 

### 1. **Loading and Preparing the Dataset**
Initially, the required dataset for training the model is downloaded and extracted from a zip file:

```python
url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")
keras.utils.get_file(filename, url)

with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")
```

### 2. **Processing Bounding Boxes**
For object detection, bounding boxes need to be processed. The code includes functions to convert box coordinates between different formats and compute the Intersection over Union (IoU) matrix, which is used for matching ground truth and anchor boxes.

- `swap_xy(boxes)`: Swaps the X and Y coordinates.
- `convert_to_xywh(boxes)`: Converts bounding boxes to the (center, width, height) format.
- `compute_iou(boxes1, boxes2)`: Calculates the IoU matrix for matching ground truth and anchor boxes.

### 3. **Generating Anchor Boxes**
An `AnchorBox` class generates anchor boxes to match the ground truth boxes. These anchors help the model predict possible object locations within an image.

```python
class AnchorBox:
    def __init__(self):
        # Define anchor scales and aspect ratios
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        # Compute anchor box dimensions for all ratios and scales
    def get_anchors(self, image_height, image_width):
        # Generate anchor boxes for an image of given dimensions
```

### 4. **Preprocessing the Data**
At this stage, images and bounding boxes are converted into a format suitable for model input. Images are resized and padded, and bounding boxes are reformatted accordingly.

```python
def preprocess_data(sample):
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)
    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id
```

### 5. **RetinaNet Model Architecture**
The RetinaNet model uses ResNet50 as a backbone network to extract features from images. The extracted features are processed using a Feature Pyramid, and class and bounding box predictions are generated through a series of convolutional layers.

```python
class RetinaNet(keras.Model):
    def __init__(self, num_classes, backbone=None, **kwargs):
        super().__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")
```

### 6. **Training the Model**
The RetinaNet model is compiled for training using the `SGD` optimizer and a custom loss function that combines Focal Loss and Smooth L1 Loss. The model is then trained on the dataset.

```python
model.compile(loss=loss_fn, optimizer=optimizer)
model.fit(train_dataset.take(100), validation_data=val_dataset.take(50), epochs=epochs)
```

### 7. **Evaluation and Inference**
After training, the model is tested on an image, where it detects objects and displays the predicted bounding boxes.

```python
image = cv2.imread(path)
detections = inference_model.predict(input_image)
visualize_detections(image, detections.nmsed_boxes[0], class_names, detections.nmsed_scores[0])
```

This code first trains the RetinaNet model and then uses it to detect objects in test images.
---
### Output of this project :

![animSeg](https://github.com/user-attachments/assets/f75a66e1-57c7-4920-ab15-b19802ad4947)


