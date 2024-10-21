import os
import random
import shutil


def split_data(data_dir, train_dir, val_dir, val_ratio=0.2):
    breeds = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for breed in breeds:
        breed_path = os.path.join(data_dir, breed)
        images = [f for f in os.listdir(breed_path) if os.path.isfile(os.path.join(breed_path, f))]
        random.shuffle(images)
        val_count = int(len(images) * val_ratio)
        train_images = images[val_count:]
        val_images = images[:val_count]

        os.makedirs(os.path.join(train_dir, breed), exist_ok=True)
        os.makedirs(os.path.join(val_dir, breed), exist_ok=True)

        for img in train_images:
            shutil.move(os.path.join(breed_path, img), os.path.join(train_dir, breed, img))

        for img in val_images:
            shutil.move(os.path.join(breed_path, img), os.path.join(val_dir, breed, img))

        os.rmdir(breed_path)

data_dir = '/Users/ayushc./Downloads/Images'
train_dir = '/Users/ayushc./Downloads/data/train'
val_dir = '/Users/ayushc./Downloads/data/validation'

split_data(data_dir, train_dir, val_dir)
