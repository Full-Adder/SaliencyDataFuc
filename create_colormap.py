import os
import numpy as np
import cv2


def create_colormap(heatmap_path, img_path, save_path, size=(112,112)):
    assert os.path.exists(heatmap_path), f"{heatmap_path} not exists"
    assert os.path.exists(img_path), f"{img_path} not exists"
    # 用cv2加载原始图像
    heatmap, img =cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE), cv2.imread(img_path)
    heatmap, img = cv2.resize(heatmap, size), cv2.resize(img, size)   
    # 使用cv2的applyColorMap函数生成热力图
    heatmap = np.uint8((heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-5) * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 将原始图像和热力图叠加
    superimposed_img = heatmap * 0.4 + img * 0.5
    # 保存叠加后的图像
    cv2.imwrite(save_path, superimposed_img)


def create_colormap_floader(heatmap_floader, img_floader, save_floader, size=(112,112)):
    assert os.path.exists(heatmap_floader), f"{heatmap_floader} not exists"
    assert os.path.exists(img_floader), f"{img_floader} not exists"

    if not os.path.exists(save_floader):
        os.makedirs(save_floader)
        print(f"create {save_floader} success")
    
    img_list = os.listdir(img_floader)
    for img_name in img_list:
        img_id = int(img_name.split('.')[0][-5:])

        heatmap_path = os.path.join(heatmap_floader, f'eyeMap_{img_id:05d}.jpg')
        img_path = os.path.join(img_floader, f'img_{img_id:05d}.jpg')
        save_path = os.path.join(save_floader, f'color_map_{img_id:05d}.jpg')

        if not os.path.exists(heatmap_path):
            print(f"{heatmap_path} not exists")
            continue
        if not os.path.exists(img_path):
            print(f"{img_path} not exists")
            continue
        
        create_colormap(heatmap_path, img_path, save_path, size)
        if os.path.exists(save_path):
            print(f"create {save_path} color_map success")

if __name__ == "__main__":
    create_colormap('eyeMap_00001.jpg', 'img_00001.jpg', 'color_map.png', (640, 480))
    # create_colormap_floader('heatmap_floader', 'img_floader', 'color_map_floader')

    
