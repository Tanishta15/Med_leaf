from fastai.vision.all import *
from fastai.callback.all import *
from fastai.metrics import *
import numpy as np
import seaborn as sns
from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.models as models

# Step 1: Define the path to your dataset
path = Path('/Users/tanishta/Desktop/College/dm_proj/Indian Medicinal Leaves Image Datasets/Medicinal Leaf dataset')

# Check if the path exists
if not path.exists():
    print(f"Error: The path {path} does not exist.")
else:
    print(f"Dataset path: {path}")

# Optional: verify images and remove broken ones
failed = verify_images(get_image_files(path))
if failed:
    print(f"Removing {len(failed)} corrupt images.")
    for f in failed:
        f.unlink()

# Step 2: Create the DataLoaders with transformations
dls = ImageDataLoaders.from_folder(
    path,
    valid_pct=0.2,
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224),
    num_workers=0  # helps debug on macOS
)

print(f'Training data size: {len(dls.train_ds)}')
print(f'Validation data size: {len(dls.valid_ds)}')

# Step 3: Create the learner with ResNet-50
learn = cnn_learner(dls, models.resnet50, metrics=accuracy)

# Step 4: Define SaveModelCallback
class SaveModelCallback(Callback):
    def __init__(self, fname):
        self.fname = fname

    def after_epoch(self):
        epoch_num = self.epoch + 1
        save_path = f'/Users/tanishta/Desktop/College/dm_proj/Trained/{self.fname}-epoch-{epoch_num}'
        self.learn.save(save_path)
        print(f'Model saved: {save_path}.pth')

# Step 5: Train the model
learn.fine_tune(6, cbs=[SaveModelCallback('resnet50-medicinal')])

# Step 6: Export the trained model
learn.export('/Users/tanishta/Desktop/College/dm_proj/ResNet50Final.pkl')

# Step 7: Load and test on a sample image
learn = load_learner('/Users/tanishta/Desktop/College/dm_proj/ResNet50Final.pkl')

test_image_path = '/Users/tanishta/Desktop/College/dm_proj/Test Images/sample.jpg'
img = PILImage.create(test_image_path)
pred_class, pred_idx, probs = learn.predict(img)
probs *= 100
print(f'Prediction: {pred_class}, Probability: {probs[pred_idx]:.4f}%')

# Step 8: Metrics and Confusion Matrix
preds, targs = learn.get_preds(dl=dls.valid)
pred_labels = preds.argmax(dim=1).numpy()
targs = targs.numpy()

f1 = f1_score(targs, pred_labels, average='macro')
acc = accuracy_score(targs, pred_labels)
precision = precision_score(targs, pred_labels, average='macro')
print(f"F1 Score: {f1:.4f}")
print(f'Validation Accuracy: {acc:.4f}')
print(f'Precision Score: {precision:.4f}')

# Confusion Matrix
class_labels = dls.vocab
cm = confusion_matrix(targs, pred_labels, labels=np.arange(len(class_labels)))
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
