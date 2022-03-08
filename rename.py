import os
import cv2
dataset_path = "D:/Anaconda/Driver monitoring system/dataset/temp/"
new_dataset_path = "D:/Anaconda/Driver monitoring system/dataset/renamed_temp/"

if not os.path.exists(new_dataset_path):
        print(f"Directory unavailable, Creating directory {new_dataset_path}")
        os.mkdir(new_dataset_path)


for cls in ["yawn"]:
    path = os.path.join(dataset_path, cls)
    new_path = os.path.join(new_dataset_path, cls)

    if not os.path.exists(new_path):
        print(f"Directory unavailable, Creating directory {new_path}")
        os.mkdir(new_path)
    else:
        print(f"Directory exists: {new_path}")

    files = os.listdir(path)
    #print(len(files))

    for file in files:
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if file.endswith('.jpeg'):
            file = file.replace('.jpeg', '.jpg')
        elif file.endswith('.png'):
            file = file.replace('.png', '.jpg')
        elif file.endswith('.JPG'):
            file = file.replace('.JPG', '.jpg')
        else:
            pass
        if file.endswith('.jpg'):
            new_img_path = os.path.join(new_path, file)
            cv2.imwrite(new_img_path, img)
            print(new_img_path + " Is saved in new directory.")