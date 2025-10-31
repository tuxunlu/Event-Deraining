from datasets import load_dataset
import os
from datasets import load_from_disk
import matplotlib.pyplot as plt

download_path = "./data"
os.makedirs(download_path, exist_ok=True) 

if not os.path.exists(download_path + "/EventRain27K"):
    ds = load_dataset("Rshnn/EventRain-27K")
    ds.save_to_disk(download_path + "/EventRain27K")

ds = load_from_disk(download_path + "/EventRain27K")
print("dataset structure: ",ds)
train_ds = ds['train']
print("train dataset length: ", len(train_ds))

example = train_ds[0]  # first sample
print(example.keys())  # show available fields

# Assuming dataset has images like 'event', 'rainy', 'clean'
plt.figure(figsize=(12,4))
for i, key in enumerate(example.keys()):
    plt.subplot(1, len(example.keys()), i+1)
    plt.imshow(example[key])
    plt.title(key)
    plt.axis('off')

plt.show()
