from fastai.vision.all import *
from fastai.callback.all import *
from fastai.vision.models import efficientnet_b0
from fastai.metrics import *
import numpy as np
import seaborn as sns
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

# Step 1: Define the path to your new dataset
path = Path('/Users/tanishta/Desktop/College/dm_proj/Indian Medicinal Leaves Image Datasets/Medicinal Leaf dataset')

# Check if the path exists
if not path.exists():
    print(f"Error: The path {path} does not exist.")
else:
    print(f"Dataset path: {path}")

# Step 2: Create the DataLoaders with transformations
dls = ImageDataLoaders.from_folder(
    path, 
    valid_pct=0.2,  # Use 20% of data for validation
    item_tfms=Resize(448),  # Resize images to 448x448
    batch_tfms=aug_transforms(size=224, max_warp=0)  # Data augmentations
)

# Print dataset details
print(f'Training data size: {len(dls.train_ds)}')
print(f'Validation data size: {len(dls.valid_ds)}')

# Step 3: Create the learner with EfficientNet architecture
learn = vision_learner(dls, efficientnet_b0, metrics=accuracy)

# Step 4: Train the model and save checkpoints
class SaveModelCallback(Callback):
    def __init__(self, fname):
        self.fname = fname

    def after_epoch(self):
        epoch_num = self.epoch + 1  # 1-based index for epoch number
        self.learn.save(f'/Users/tanishta/Desktop/College/dm_proj/Trained/{self.fname}-epoch-{epoch_num}')
        print(f'Model saved: {self.fname}-epoch-{epoch_num}.pth')

learn.fine_tune(6, cbs=[SaveModelCallback('efficientnet-medicinal')])

# Step 5: Save the trained model
learn.export('/Users/your_username/Python Projects/MedicinalLeafDataset/EfficientNetFinal.pkl')

# Step 6: Load the model for testing
learn = load_learner('/Users/your_username/Python Projects/MedicinalLeafDataset/EfficientNetFinal.pkl')

# Step 7: Load a new image and predict its class
test_image_path = '/Users/your_username/Python Projects/MedicinalLeafDataset/Test Images/sample.jpg'
img = PILImage.create(test_image_path)
pred_class, pred_idx, probs = learn.predict(img)
probs *= 100
print(f'Prediction: {pred_class}, Probability: {probs[pred_idx]:.4f}%')

# Step 8: Calculating metrics and confusion matrix
preds, targs = learn.get_preds(dl=dls.valid)
pred_labels = preds.argmax(dim=1).numpy()
targs = targs.numpy()

f1 = f1_score(targs, pred_labels, average='macro')
acc = accuracy_score(targs, pred_labels)
precision = precision_score(targs, pred_labels, average='macro')
print(f"F1 Score: {f1}")
print(f'Validation Accuracy: {acc:.4f}')
print(f'Precision Score: {precision:.4f}')

# Confusion Matrix
class_labels = dls.vocab  # Automatically extract class names
cm = confusion_matrix(targs, pred_labels, labels=np.arange(len(class_labels)))

# Convert confusion matrix to a DataFrame for visualization
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()