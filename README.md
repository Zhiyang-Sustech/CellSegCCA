# π-CellSeg-CCA
π-CellSeg-CCA is a pathological image analysis algorithm based on Mask R-CNN and ResNet-18, to enable automated annotation of CCA and normal bile ducts regions for LMD.

## Running the code
### Segmentation
#### Train
##### Data Format
For training, all CCA/normal bile duct images should be saved as standard images files, png is recommended. The masks should be saved as mat files. The corresponding image and mask file names need to be the same
Before running the code, all images must be divided into training and tset set, then saved in the following structure:

- <Data_path>
  - train
    - cancer
      - image
      - masks
    - normal
      - image
      - masks
  - test
    - cancer
      - image
      - masks
    - normal
      - image
      - masks

##### Usage
1. Modify train.param
2. Run main.py

The trained model will be output in floder 'logs' as an h5 file.

#### Test

