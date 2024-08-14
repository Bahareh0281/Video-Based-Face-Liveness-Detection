# Anti-Spoofing Algorithm Project

This repository contains the code and resources for the project titled **"Designing Classic and Deep Models for Anti-Spoofing Algorithm"**, supervised by Dr. Mohammadi as part of the Computer Vision course. The project involves building and evaluating models to detect face spoofing using both classical and deep learning approaches.

## Dataset

The dataset used in this project is a combination of the 
[FASD-CASIA](https://paperswithcode.com/dataset/casia-fasd) dataset and additional [anti-spoofing-face-fake](https://universe.roboflow.com/huu-thinh-muem8/anti-spoofing-face-fake/dataset/1/download) images. The combined dataset includes images and videos labeled as either `real` or `fake`.

### Dataset Preparation

1. **Uploading and Loading Data:**  
   The dataset is uploaded to the [Hugging Face](https://huggingface.co/) platform and loaded into Google Colab for processing.

2. **Labeling:**  
   Image files are labeled based on their filenames:
   - `***_fake.jpg` → Label: 0
   - `***_real.jpg` → Label: 1

3. **Frame Extraction:**  
   For the testing phase, frames are extracted from videos. A random frame from each video is saved with a label indicating whether it is `real` or `fake`.

## Models

### Deep Learning Models

1. **ResNet-50:**
   - Pretrained on ImageNet.
   - Modified by replacing the fully connected (FC) layer with a Global Average Pooling (GAP) layer, followed by a dense layer with 1024 neurons and a final FC layer with 2 neurons for classification.

2. **ViT-Base-16/224 (Google):**
   - Pretrained Vision Transformer model.
   - Fine-tuned using the project dataset.

#### Training Process
- Data preprocessing includes resizing images to `224x224` and normalizing pixel values.
- Labels are converted to one-hot encoded format.
- The models are trained on the training set and evaluated on the test set, including both random and cropped frames.

### Classical Machine Learning Models

1. **CNN:**
   - A simple CNN model with several convolutional layers followed by fully connected layers.
   - Extracts features such as frequency, Local Binary Patterns (LBP), depth, and statistical features.

2. **InceptionV3:**
   - Another model tested with an input size of `75x75`.
   - Evaluated similarly to the other models.

## Evaluation

### ResNet-50 and ViT Models

- **Evaluation Metrics:**
  - Accuracy on random frames.
  - Accuracy on cropped frames using MTCNN for face detection and cropping.

### CNN and InceptionV3 Models

- **Evaluation Metrics:**
  - Feature extraction on a subset of the dataset.
  - Accuracy on raw and cropped frames.

## Results

- The ResNet-50 model achieved an accuracy of approximately 72% on cropped test images.
- The ViT model underwent fine-tuning for improved performance.
- The CNN model reached an accuracy of 65% on raw test frames.
- InceptionV3 was also evaluated and produced notable results, which are detailed in the CSV files.

## File Structure

- **`/data/`**: Contains the dataset images and videos.
- **`/documents/`**: project's report (Farsi)
- **`/notebooks/`**: Jupyter notebooks for data preprocessing, model training, and evaluation.
- **`/results/`**: CSV files with predictions and accuracy metrics for the models.

## How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Bahareh0281/Video-Based-Face-Liveness-Detection.git
   cd anti-spoofing-algorithm
   ```

2. **Install dependencies:**
   Ensure you have Python 3 and the required libraries installed.

3. **Run the notebooks:**
   Navigate to the `/notebooks/` directory and open the relevant notebook in Google Colab or Jupyter.

4. **Evaluate the models:**
   Follow the instructions in the notebooks to train and evaluate the models using your dataset.

## Acknowledgements

This project was conducted under the supervision of **Dr. Mohammadi** as part of the Computer Vision course.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.