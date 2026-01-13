import os, json

# Path to dataset
dataset_path = "dataset/"

# Get folder names in alphabetical order (same as Keras uses)
labels = sorted(os.listdir(dataset_path))

# Save labels
with open("labels.json", "w") as f:
    json.dump(labels, f)

print("✅ Labels saved:", labels)
