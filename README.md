# Animal recognition with a Vision Transformer

Image classification of animal photos into 90 species with a Vision Transformer (ViT). I fine-tuned the pretrained `google/vit-base-patch16-224-in21k` model on Kaggle's [Animal Image Dataset (90 different animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals), using PyTorch and Hugging Face Transformers.

## Where the files are

The training script is in this repository: [`animal_recognition.py`](animal_recognition.py). The dataset (about 690 MB) and the trained model (about 320 MB) are too large for GitHub, so they are in a Google Drive folder:

https://drive.google.com/drive/folders/1-9O1tVvN1U1fdPU0BB-PBp1eu0fyBncv?usp=sharing

To train the model again, run the script on Kaggle with the dataset attached (it reads it from `/kaggle/input/animal-image-dataset-90-different-animals`). Set `image_path` at the end of the script to try the model on your own picture.

## How it works

The model takes 224 × 224 images cut into 16 × 16 patches and reads the patches as a sequence. Starting from weights pretrained on ImageNet-21k, I fine-tuned it on the animal images (transfer learning).

Training setup: a random 80/20 split between training and test images, 4 epochs, batch size 16, learning rate 2e-4, on a Kaggle GPU. The script reports accuracy and macro F1-score on the test set, then exports the model to TorchScript (`vit_model.pt`).

## Technologies

Python, PyTorch, Hugging Face Transformers, torchvision for image preprocessing, scikit-learn for the evaluation metrics, Pillow and Kaggle.

## Author

Chems Eddine Adel Kebaily, Master's student in Signal and Image Processing at ULCO (Université du Littoral Côte d'Opale). [Portfolio](https://chemseddine-kebaily.netlify.app), [LinkedIn](https://www.linkedin.com/in/chemseddineadel-keb).
