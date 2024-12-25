import os
import scipy.io as sio
import numpy as np
import skimage.io
import cv2
import copy
#from tensorflow.keras.optimizers import Adam
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch
import random
import CTool
import pandas as pd
from CFunction import CFunctionCaculateResult

def extract_bboxes(mask, num):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([num, 4], dtype=np.int32)
    for i in range(1, num + 1):
        m = np.zeros(mask.shape)
        m[mask == i] = 1
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i - 1] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def image2patches(image_path, mat_path, output_path):
    mats = os.listdir(mat_path)
    for mat in mats:
        base_name = mat.split(".mat")[0]
        print(base_name)
        mask = sio.loadmat(os.path.join(mat_path, mat))
        mask = (mask['mask']).astype("int32")
        num = np.max(mask)
        bbox = extract_bboxes(mask, num)        # bbox:x1 y1 x2 y2
        image = skimage.io.imread(os.path.join(image_path, "%s.png" % base_name))
        image = image[:, :, :3]
        for i in range(bbox.shape[0]):
            crop_img = image[bbox[i][0]:bbox[i][2], bbox[i][1]:bbox[i][3], :]
            cv2.imwrite("%s/%s_%s.png" % (output_path, base_name, str(i)), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))


def TrainModel(data_dir, lr, is_fine_tuning, batch_size, optimizer, epochs):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    nn.init.xavier_uniform_(model.fc.weight)

    data_transforms = {
        'train': transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 创建训练集和验证集的dataloaders
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                        ['train', 'val']}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)

    # 是否采用微调
    if is_fine_tuning:
        param = []
        params_1x = [param for name, param in model.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        param.append({'params': params_1x})
        param.append({'params': model.fc.parameters(), 'lr': lr})
        #optimizer = GetOptim(optimizer, param, lr / 10)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        #optimizer = GetOptim(optimizer, model.parameters(), lr)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    val_acc_history = []
    loss_his = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)  # inputs.size(0) = batch_size

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            if phase == 'train':
                loss_his['train'].append(epoch_loss)
            else:
                loss_his['val'].append(epoch_loss)
            epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, loss_his

def GetOptim(optim_name, param, learning_rate):
    if optim_name == "SGD":
        return optim.SGD(param, lr=learning_rate, weight_decay=0.1, momentum=0)
    elif optim_name == "Adam":
        return optim.Adam(param, lr=learning_rate, weight_decay=0)
    else:
        return optim.RMSprop(param, lr=learning_rate, weight_decay=0, momentum=0)

def TestModel(data_dir, model_path, output_path, batch_size = 32):
    # 读取模型
    model_ft = torch.load(model_path)

    # 加载测试集
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataloaders = torch.utils.data.DataLoader(datasets.ImageFolder(
        data_dir, transform=data_transforms), batch_size=batch_size)

    # 获取输入图片的病历号
    list_patient_index = []
    for data in dataloaders.dataset.imgs:
        temp = data[0].split("\\")
        list_patient_index.append(temp[len(temp) - 1].strip())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    model_ft.eval()

    # 测试模型
    list_output = []
    list_label = []
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)  # tensor (batchsize,1, width, height)
        labels = labels.to(device)  # tensor (batchsize)

        list_label.append(labels)
        with torch.set_grad_enabled(False):
            list_output.append(model_ft(inputs))  # tensor(batchsize, 2)

    # 对所有batch_size的输出在第一维度上拼接
    label = torch.cat(list_label, 0)
    output = torch.cat(list_output, 0)

    m = nn.Softmax(dim=1)
    output = m(output)

    _, preds = torch.max(output, 1)  # tensor(num of data,)

    # 统计测试结果
    result = CFunctionCaculateResult(output_path, label, preds, output[:, 1])
    print(result)
    # 保存测试性能指标
    df = pd.DataFrame()
    df_add = pd.DataFrame.from_dict(result.values()).T
    df_add.columns = result.keys()
    df = pd.concat([df, df_add])
    df.to_excel(os.path.join(output_path, "test_results.xlsx"), index=False)

    # 保存预测结果
    df = pd.DataFrame(list_patient_index, columns=['patient'])
    df_pred = pd.DataFrame(preds, columns=['pred'])
    df = pd.concat([df, df_pred], axis=1)
    df['true'] = label
    df.insert(2, 'positive prob', output[:, 1])
    df.to_excel(os.path.join(output_path, "test_pred.xlsx"), index=False)


if __name__ == '__main__':
    is_preprocess = 0
    is_train = 0
    if is_preprocess:
        print("image to little patch")
        image_path = r'D:\DVPszy\data\outside\tcga_test\cancer'
        mat_path = r'D:\DVPszy\data\result\tcga_cancer\mat'
        output_path = r'D:\DVPszy\data\outside\tcga_test\small\cancer'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 基于bbox将分割对象切为小图像
        image2patches(image_path, mat_path, output_path)

    elif not is_preprocess and is_train:
        print("train model")

        data_dir = r"D:\DVPszy\data\CC_doctor\New\small_pictures_for_classification\Hu\train_patchs_fix2\patchs"
        output_path = r'D:\DVPszy\data\classification_mod\Hu_fix22'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        lr = 0.01
        is_fine_tuning = False
        optimizer = 'Adam'
        epochs = 40
        batch_size = 16



        model_ft, val_acc, loss = TrainModel(data_dir, lr, is_fine_tuning, batch_size, optimizer, epochs)
        # 保存最佳模型
        torch.save(model_ft, os.path.join(output_path, 'best_model.pth'))
        # 画图
        CTool.DrawValAcc(val_acc, epochs, output_path)
        CTool.DrawTrainValLoss(loss, epochs, output_path)
    else:
        print("test model")
        data_dir = r'D:\DVPszy\data\outside\SYSMH-S-20230918\small_test2'
        model_path = r"D:\DVPszy\data\classification_mod\Hu_fix22\best_model.pth"
        output_path = r'D:\DVPszy\data\outside\SYSMH-S-20230918\Hu_fix2_test_in_sec'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        TestModel(data_dir, model_path, output_path)


