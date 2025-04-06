import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from fastai.vision.all import *
from fastai.callback.all import *
from fastai.metrics import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.models as models
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# Step 1: Define the path to your dataset
path = Path('/Users/tanishta/Desktop/College/dm_proj/Indian Medicinal Leaves Image Datasets/Medicinal Leaf dataset')

# Check if the path exists
if not path.exists():
    raise FileNotFoundError(f"Error: The path {path} does not exist.")
print(f"✅ Dataset path found: {path}")

# Optional: Remove any corrupt image files
def remove_corrupt_images(path):
    for f in get_image_files(path):
        try:
            _ = PILImage.create(f)
        except Exception as e:
            print(f"Removing corrupt image: {f.name} ({e})")
            f.unlink()

remove_corrupt_images(path)

# Step 2: Create the DataLoaders with transformations
dls = ImageDataLoaders.from_folder(
    path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224),
    bs=32
)

print(f'📊 Training samples: {len(dls.train_ds)}')
print(f'📊 Validation samples: {len(dls.valid_ds)}')

# Step 3: Create the learner with DenseNet-121
learn = cnn_learner(dls, models.densenet121, metrics=accuracy)

# Step 4: SaveModelCallback
class SaveModelCallback(Callback):
    def __init__(self, fname): self.fname = fname

    def after_epoch(self):
        epoch_num = self.epoch + 1
        save_path = f'/Users/tanishta/Desktop/College/dm_proj/Trained/{self.fname}-epoch-{epoch_num}'
        self.learn.save(save_path)
        print(f'💾 Model saved at epoch {epoch_num}: {save_path}.pth')

# Step 5: Train the model
learn.fine_tune(6, cbs=[SaveModelCallback('densenet121-medicinal')])

# Step 6: Export the trained model
learn.export('/Users/tanishta/Desktop/College/dm_proj/DenseNet121Final.pkl')
print("✅ Model exported successfully.")

# Step 7: Load and test on a sample image
learn = load_learner('/Users/tanishta/Desktop/College/dm_proj/DenseNet121Final.pkl')

test_image_path = Path('/Users/tanishta/Desktop/College/dm_proj/Test Images/sample.jpg')
if test_image_path.exists():
    img = PILImage.create(test_image_path)
    pred_class, pred_idx, probs = learn.predict(img)
    print(f'🔍 Prediction: {pred_class} ({probs[pred_idx]*100:.2f}%)')
else:
    print(f"⚠️ Test image not found at {test_image_path}")

# Step 8: Evaluate model
preds, targs = learn.get_preds(dl=dls.valid)
pred_labels = preds.argmax(dim=1).numpy()
targs = targs.numpy()

f1 = f1_score(targs, pred_labels, average='macro')
acc = accuracy_score(targs, pred_labels)
precision = precision_score(targs, pred_labels, average='macro')

print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Precision: {precision:.4f}")

# Confusion Matrix
class_labels = dls.vocab
cm = confusion_matrix(targs, pred_labels)
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()
