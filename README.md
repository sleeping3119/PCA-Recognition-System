# PCA Recognition System

A Python-based application that leverages **Principal Component Analysis (PCA)** for advanced image recognition and classification. This system can distinguish faces, detect attributes like glasses, and recognize emotional expressions by analyzing the underlying patterns in image data.

---

## Key Features

| Feature               | Description                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| **Face Identification** | Identifies individuals by matching test images against a trained dataset of faces.                     |
| **Attribute Detection** | Discerns specific attributes, such as the presence of glasses or the lighting conditions (shade).        |
| **Emotion Recognition** | Classifies a range of facial expressions including happy, sad, sleepy, and surprised.                    |
| **Visual Feedback** | Displays each test image with its best-matched category and the associated confidence score (loss value). |

---

## How It Works

The core of this project is **Principal Component Analysis (PCA)**, a powerful dimensionality reduction technique used here to create a unique "feature space" for each image category (e.g., `happy`, `person_1`, `glasses`).

1.  **Training:** For each category in the `images` directory, the system learns a **PCA basis** (a set of principal components). This basis represents the most important features of that category.
2.  **Testing:** When a new image from the `test` folder is introduced, the system projects it onto the PCA basis of every trained category.
3.  **Classification:** It then calculates the **reconstruction error** (or "loss") for each projection. The image is classified as belonging to the category that yields the **lowest loss**, as this indicates the most accurate reconstruction and therefore the best match.

The specific feature being tested—whether it's identifying a person, detecting glasses, or recognizing an emotion—is determined by your selection from a menu when the script is run.

---

## Dataset Re

The project's accuracy is highly dependent on how the data is structured. The datasets are organized into two main folders: `images` for training and `test` for evaluation.

-   `images/`: Contains individual subdirectories, where each subdirectory represents a single class (e.g., `person_1`, `happy`, `glasses`).
-   `test/`: Contains the images that the system will attempt to classify.

Here is a simplified view of the required structure:

```
.
├── images/
│   ├── category_1/      # e.g., 'glasses'
│   ├── category_2/      # e.g., 'noglasses'
│   └── ...
└── test/
    ├── test_image_1.jpg
    └── ...
```

---

## Installation & Usage

### Installation

To get started, clone the repository and install the necessary packages.

**If you dont have dataset according to above requirements download it from [Releases Page](https://github.com/sleeping3119/PCA-Recognition-System/releases)**
```bash
# Clone the repository
git clone https://github.com/sleeping3119/PCA-Recognition-System.git
cd PCA-Recognition-System

# Install dependencies
pip install -r requirements.txt
```

### Running the System

Execute the `main.py` script from your terminal.

```bash
python main.py
```

You will be presented with a menu to choose the recognition task you wish to perform. The application will then process the images in the `test` folder and display the classification results.
