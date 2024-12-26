# π-CellSeg-CCA
π-CellSeg-CCA is a pathological image analysis algorithm based on Mask R-CNN and ResNet-18, to enable automated annotation of CCA and normal bile ducts regions for LMD.

## Running the code

### Segmentation
#### Train
##### Data Format
For training, all CCA/normal bile ducts images should be saved as standard images files, png is recommended. The masks should be saved as mat files. The corresponding image and mask file names need to be the same
Before running the code, all images must be divided into training and validation set, then saved in the following structure:

- <Data_path>
  - train
    - cancer
      - image
      - masks
    - normal
      - image
      - masks
  - val
    - cancer
      - image
      - masks
    - normal
      - image
      - masks

##### Usage
1. Modify train.param
  The options including:
  PATH_TRAIN_DATA
  PATH_VAL_DATA
  IMAGE_MIN_DIM
  IMAGE_MAX_DIM
  LEARNING_RATE
  GPU_COUNT
  IMAGES_PER_GPU
  STEPS_PER_EPOCH
  VALIDATION_STEPS
  RPN_TRAIN_ANCHORS_PER_IMAGE
  RPN_NMS_THRESHOLD
  TRAIN_ROIS_PER_IMAGE
  MAX_GT_INSTANCES
  DETECTION_MAX_INSTANCES
  DETECTION_NMS_THRESHOLD

2. Run main.py

The trained model will be output in floder 'logs' as an h5 file.

#### Test
##### Data Format
The data format for test is similar with training,  all images must be saved in the following structure:
- <Data_path>
  - test
    - cancer
      - image
      - masks
    - normal
      - image
      - masks

The binary classification and masks are only used for evaluation and are not required in the segmentation of real samples.

##### Usage
1. Modify test.param
  The options including:
  TRAIN_SCAN
  TEST_SCAN
  IMAGE_RESIZE_MODE
  ZOOM
  ASPECT_RATIO
  MIN_ENLARGE
  IMAGE_MIN_DIM
  IMAGE_MAX_DIM
  GPU_COUNT
  IMAGES_PER_GPU
  RPN_NMS_THRESHOLD
  DETECTION_MIN_CONFIDENCE
  DETECTION_NMS_THRESHOLD
  DETECTION_MAX_INSTANCES

2. Run main.py

The results include annotated overlay images and masks.

### Classfication

All CCA/normal bile ducts cell clusters in annotated images must be extract to little patches first.
Set "is_preprocess = 1, is_train = 0" in cells_classification.py, the results output by segmentation model can be extract into little patches, each patch contains one CCA/normal BD cell cluster.

#### Training
##### Data Format

Before training, all images must be divided into training and validation set, then saved in the following structure:

- <Data_path>
  - train
    - 0_normal
    - 1_cancer
  - val
    - 0_normal
    - 1_cancer

##### Usage

1. Set "is_preprocess = 1, is_train = 0" in cells_classification.py
2. Set the data path
3. Run cells_classification.py

The trained model will be output as an pth file.

#### Test
##### Data Format

The data format for test is similar with training,  all images must be saved in the following structure:

- <Data_path>
  - test
    - 0_normal
    - 1_cancer

The binary classification are only used for evaluation and are not required in the segmentation of real samples.

##### Usage

1. Set "is_preprocess = 0, is_train = 0" in cells_classification.py
2. Set the data path
3. Run cells_classification.py

The results, including predictions and positive probability, will be output in tables.


  



