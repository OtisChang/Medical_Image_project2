from pathlib import Path
import os
import random

random.seed(123)


def move(src, dst, size=0.1):
    fileList = os.listdir(src[0])
    random.shuffle(fileList)
    batch = round(1 / size)
    print("total images: , batch= ", len(fileList), batch)
    for i, path in enumerate(src):
        for j, item in enumerate(fileList):
            print("image name", item)
            if j % batch == 0:
                Path(src[i]+item).rename(dst[i]+item)


print("---------------------------------")
move(["./data/imgs/", "./data/masks/"], ["./test/imgs/", "./test/masks/"], 0.2)
