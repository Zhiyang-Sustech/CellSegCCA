import os
from scipy import ndimage
import numpy as np
import scipy.io as sio
import math
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import ElementTree
from xml.dom import minidom

# 该函数使xml文件更加美观，也就是换行和缩进
def prettyXml(element, indent, newline, level=0):
    '''
    参数:
    elemnt为传进来的Elment类;
    indent用于缩进;
    newline用于换行;
    '''
    # 判断element是否有子元素
    if element:
        # 如果element的text没有内容
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # 此处两行如果把注释去掉，Element的text也会另起一行
    # else:
    # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将elemnt转成list
    for subelement in temp:
        # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
            # 对子元素进行递归操作
        prettyXml(subelement, indent, newline, level=level + 1)

def one_side(dict_boundary, optimized_boundary, forward, ascend):
    last_x = optimized_boundary[-1][0]
    x_index = list(dict_boundary.keys())  # 从小到大排序
    last_x_index = x_index.index(last_x)

    # x从小到大遍历
    if forward:
        first_x = last_x_index + 1
        ys = dict_boundary[x_index[first_x]]
        # 更换方向后添加的第一排坐标
        if ascend:
            for j in range(len(ys)):
                if math.fabs(ys[j]- optimized_boundary[-1][1]) <= 1 and ys[j] >= optimized_boundary[-1][1]:
                    optimized_boundary.append([x_index[first_x], ys[j]])
        else:
            for j in range(len(ys)-1,-1,-1):
                if math.fabs(ys[j]- optimized_boundary[-1][1]) <= 1 and ys[j] <= optimized_boundary[-1][1]:
                    optimized_boundary.append([x_index[first_x], ys[j]])
        for i in range(first_x + 1, len(x_index)):
            is_continued = False
            ys = dict_boundary[x_index[i]]
            # 若ys中存在与上一个坐标先沟通的y，则先从该位置开始
            begin = 0
            for j in range(len(ys)):
                if ys[j]==optimized_boundary[-1][1]:
                    begin = j
                    break
            for j in range(begin, len(ys)):
                if math.fabs(ys[j]- optimized_boundary[-1][1]) <= 1:
                    is_continued = True
                    optimized_boundary.append([x_index[i], ys[j]])
            # x方向上不存在连续的坐标时
            if not is_continued:
                break
            # 为避免路径往Y变小的方向走错过坐标点，同一行上需要反向遍历是否符合规则的坐标
            # 需要注意不能反向重复添加，导致出现路径切半现象！！！
            # 先定位到最后最后添加的坐标，再判断是否需要反向遍历
            for j in range(len(ys)-1, -1, -1):
                if ys[j] == optimized_boundary[-1][1]:
                    break
            if j > 0 and ys[j-1] != optimized_boundary[-2][1]:
                for t in range(j-1, -1, -1):
                    if math.fabs(ys[t]- optimized_boundary[-1][1]) <= 1:
                        optimized_boundary.append([x_index[i], ys[t]])
                    else:
                        break
    # x从大到小遍历
    else:
        first_x = last_x_index - 1
        ys = dict_boundary[x_index[first_x]]
        # 更换方向后添加的第一排坐标
        if ascend:
            for j in range(len(ys)):
                if math.fabs(ys[j]- optimized_boundary[-1][1]) <= 1 and ys[j] >= optimized_boundary[-1][1]:
                    optimized_boundary.append([x_index[first_x], ys[j]])
        else:
            for j in range(len(ys) - 1, -1, -1):
                if math.fabs(ys[j]- optimized_boundary[-1][1]) <= 1 and ys[j] <= optimized_boundary[-1][1]:
                    optimized_boundary.append([x_index[first_x], ys[j]])
        for i in range(first_x - 1, 0, -1):      # 最小行值取1，不与最开始的第一行重合
            is_continued = False
            ys = dict_boundary[x_index[i]]
            # 若ys中存在与上一个坐标先沟通的y，则先从该位置开始
            begin = 0
            for j in range(len(ys)):
                if ys[j] == optimized_boundary[-1][1]:
                    begin = j
                    break
            for j in range(begin, len(ys)):
                if math.fabs(ys[j]- optimized_boundary[-1][1]) <= 1:
                    is_continued = True
                    optimized_boundary.append([x_index[i], ys[j]])
            # x方向上不存在连续的坐标时
            if not is_continued:
                break
            # 为避免路径往Y变小的方向走错过坐标点，同一行上需要反向遍历是否符合规则的坐标
            # 需要注意不能反向重复添加，导致出现路径切半现象！！！
            # 先定位到最后最后添加的坐标，再判断是否需要反向遍历
            for j in range(len(ys)-1, -1, -1):
                if ys[j] == optimized_boundary[-1][1]:
                    break
            if j > 0 and ys[j-1] != optimized_boundary[-2][1]:
                for t in range(j-1, -1, -1):
                    if math.fabs(ys[t]- optimized_boundary[-1][1]) <= 1:
                        optimized_boundary.append([x_index[i], ys[t]])
                    else:
                        break

    return optimized_boundary

def optimized_path(boundary):
    # boundary转为字典
    dict_boundary = {}
    for i in range(boundary.shape[0]):
        if boundary[i,0] not in dict_boundary:
            dict_boundary[boundary[i,0]] = [boundary[i,1]]
        else:
            dict_boundary[boundary[i, 0]].append(boundary[i,1])

    x_index = list(dict_boundary.keys())    #从小到大排序

    optimized_boundary = []
    # 从x最小行开始，顺时针遍历
    # 添加最上一排边界的左边全部连续值
    optimized_boundary.append([x_index[0], dict_boundary[x_index[0]][0]])
    for item in dict_boundary[x_index[0]][1:]:
        if math.fabs(item- optimized_boundary[-1][1]) <= 1:
            optimized_boundary.append([x_index[0], item])
        else:
            break

    forward = True
    ascend = True
    while len(optimized_boundary) < boundary.shape[0]:
        print("hh")
        optimized_boundary = one_side(dict_boundary, optimized_boundary, forward, ascend)
        forward = not forward
        if optimized_boundary[-1][1] > optimized_boundary[-2][1]:
            ascend = True
        else:
            ascend = False
    if len(optimized_boundary) > boundary.shape[0]:
        print("优化后轮廓长度：",len(optimized_boundary), "；实际轮廓长度：",  boundary.shape[0])
    return optimized_boundary

def get_boundary(mask, i):
    mask_one = np.zeros(mask.shape, dtype=int)
    mask_one[mask == i] = 1
    # 向往膨胀N轮，如果mask触碰边界则去除，其边界为膨胀N+1轮
    boundary_1 = ndimage.binary_dilation(mask_one, iterations=2).astype(mask_one.dtype)
    index = np.where(boundary_1 == 1)
    if np.any(np.isin(index[0], [0, mask_one.shape[0] - 1])) or np.any(np.isin(index[1], [0, mask_one.shape[1] - 1])):
        return False, 0, None, None

    outer_border = ndimage.binary_dilation(mask_one, iterations=3).astype(mask_one.dtype) - ndimage.binary_dilation(mask_one, iterations=2).astype(mask_one.dtype)
    pixel_count = np.sum(outer_border)
    boundary = np.transpose(np.nonzero(outer_border))
    boundary = optimized_path(boundary)
    class_id = 'A'
    return True, pixel_count, boundary, class_id

def create(output_dir, mat_path, base_name, global_coordinates, calibration_points):
    mask = sio.loadmat(mat_path)

    mask = (mask['mask']).astype("int32")
    num = np.max(mask)
    # num=100
    print("图像中一共检测出细胞核数量：", num)
    CalibrationData = Element('ImageData')

    GlobalCoordinates = SubElement(CalibrationData, 'GlobalCoordinates')
    GlobalCoordinates.text = str(global_coordinates)

    CalibrationPointX_1 = SubElement(CalibrationData, 'X_CalibrationPoint_1')
    CalibrationPointX_1.text = str(calibration_points[0])
    CalibrationPointY_1 = SubElement(CalibrationData, 'Y_CalibrationPoint_1')
    CalibrationPointY_1.text = str(calibration_points[1])
    # CalibrationFocusPosition_1 = SubElement(CalibrationData, 'CalibrationFocusPosition_1')
    # CalibrationFocusPosition_1.text = str(CalibrationFocusPositions[0])

    CalibrationPointX_2 = SubElement(CalibrationData, 'X_CalibrationPoint_2')
    CalibrationPointX_2.text = str(calibration_points[2])
    CalibrationPointY_2 = SubElement(CalibrationData, 'Y_CalibrationPoint_2')
    CalibrationPointY_2.text = str(calibration_points[3])
    # CalibrationFocusPosition_2 = SubElement(CalibrationData, 'CalibrationFocusPosition_2')
    # CalibrationFocusPosition_2.text = str(CalibrationFocusPositions[1])

    CalibrationPointX_3 = SubElement(CalibrationData, 'X_CalibrationPoint_3')
    CalibrationPointX_3.text = str(calibration_points[4])

    CalibrationPointY_3 = SubElement(CalibrationData, 'Y_CalibrationPoint_3')
    CalibrationPointY_3.text = str(calibration_points[5])
    # CalibrationFocusPosition_3 = SubElement(CalibrationData, 'CalibrationFocusPosition_3')
    # CalibrationFocusPosition_3.text = str(CalibrationFocusPositions[2])

    ShapeCount = SubElement(CalibrationData, 'ShapeCount')
    ShapeCount.text = str(num)
    real_count = 0
    for i in range(1, num + 1):
        print("no:", i)
        if i == 109 or i == 84:
            continue
        flag, pixel_count, boundary, class_id = get_boundary(mask, i)

        if not flag:
            print("轮廓贴近边界")
            continue
        # if real_count == 408:
        #     print("实际索引：", i)
        #     exit(0)
        real_count += 1
        Shape_Index = SubElement(CalibrationData, 'Shape_' + str(real_count))
        Pixel_Count = SubElement(Shape_Index, 'PointCount')
        Pixel_Count.text = str(pixel_count)
        Class_Id = SubElement(Shape_Index, 'CapID')
        Class_Id.text = class_id
        # 0列为y轴，1列为x轴
        # 并将坐标进行缩放和位移
        # 20倍图像中，x轴放大3.22倍，增加1055355，y轴放大3.26倍，增加462408
        # 40倍图像中，x轴放大1.624倍，增加1056981，y轴放大1.624，增加463447
        boundary = np.array(boundary)
        boundary[:,0] = boundary[:, 0]*1.624 + 463447
        boundary[:,1] = boundary[:, 1]*1.624 + 1056981
        boundary = boundary.astype(int)
        for j in range(pixel_count):
            X_Point = SubElement(Shape_Index, 'X_' + str(j + 1))
            X_Point.text = str(boundary[j,1])
            Y_Point = SubElement(Shape_Index, 'Y_'+ str(j + 1))
            Y_Point.text = str(boundary[j,0])

    ShapeCount.text = str(real_count)
    tree = ElementTree(CalibrationData)
    root = tree.getroot()
    prettyXml(root, '\t', '\n')

    # write out xml data
    tree.write(os.path.join(output_dir, base_name + '.xml'), encoding='utf-8', xml_declaration=True)
    # tree.write('hh.xml', encoding='utf-8', xml_declaration=True)
    # with open('hh.xml', 'w') as f:  # 保存文件为sector
    #     tree.writexml(f, addindent='  ')

mat_dir = "D:\DVPszy\data\\result\mask-rcnn-test\\train20230405T1251_10_20x\mat"
output_dir = "./"
mats = os.listdir(mat_dir)
# 20: 177,1167,1690,1050,1723,80
# 40: 207,1153,1727,1095,1735,29
calibration_points = [1057079, 465493, 1059556, 465396, 1059568, 463663]
global_coordinates = 1
for mat in mats:
    base_name = mat.split(".mat")[0]
    create(output_dir, os.path.join(mat_dir, mat), base_name, global_coordinates, calibration_points)
