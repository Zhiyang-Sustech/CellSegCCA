# π-CellSeg-CCA
π-CellSeg-CCA is a pathological image analysis algorithm based on Mask R-CNN and ResNet-18, to enable automated annotation of CCA and normal bile ducts regions for LMD.
π-CellSeg-CCA comprised a segmentation model and a classification model. Mask R-CNN with a discrete cosine transform (DCT) module integrated into the segmentation branch, termed Mask R-CNN-DCT, was employed in the segmentation task. For classification task, classic deep learning network ResNet-18 trained with the predictions of Mask R-CNN-DCT was applied to classify and predict normal and cancer cell clusters. 

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

2. Run main_train.py

The trained model will be output in floder 'logs' as an h5 file, along with the loss during training.

Examples (the loss during training):
![](/example/seg_loss.png)

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

2. Run main_test.py

The results include annotated overlay images and masks.

Examples (annotated overlay images and masks segmented by Mask R-CNN and improved Mask R-CNN-DCT):
![](/example/seg_example.png)

### Classfication

All CCA/normal bile ducts cell clusters in annotated images must be extract to little patches first.
Set "is_preprocess = 1, is_train = 0" in cells_classification.py, the results output by segmentation model can be extract into little patches, each patch contains one CCA/normal BD cell cluster.

Examples (extracted little patches):

![](/example/patch_example.png)

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

The trained model will be output as an pth file, along with the loss on training and validation set and the accuracy on validation set.

Examples (the loss on training, validation set and the accuracy on validation set):

![](/example/class_train_loss.png)

![](/example/class_val_acc.png)

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

Examples (patches and corresponding positive probability):

![](/example/class_example.png)

### Transform to XML files
After segmentated and annotated, the output masks are needed to be transformed for subsequent laser microdissection
Transform2LCM.py is prepared for transformation, set the path of masks and run the code, the transformation would be finished soon.
The output XML files stored dissection trajectory could be imported into laser microdissection system and guide dissection task.

Examples (XML file, images before and after trajectory import):

- CCA

[XML file](/example/example_CCA_XML.xml)

![](/example/example_CCA_before_import.JPG)

![](/example/example_CCA_after_import.JPG)

- Normal BD

[XML file](/example/example_NormalBD_XML.xml)

![](/example/example_NormalBD_before_import.JPG)

![](/example/example_NormalBD_after_import.JPG)
