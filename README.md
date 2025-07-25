# Ontranet

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License](https://img.shields.io/github/license/maxardito/ontranet)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-red?logo=pytorch)
![Built with Nix](https://img.shields.io/badge/Built%20with-Nix-5277C3?logo=nixos)

An onset-based transformer model for audio source classification

This README contains instructions for reproducibility. For an in depth tour of the model, see the [about](https://github.com/maxardito/ontranet/blob/main/docs/about.md) file in the `docs` directory.

## 🖥️ Installation

### OSX/Linux

Make sure you have ffmpeg and git lfs installed:

```
brew install git-lfs ffmpeg
git lfs install
```

### NixOs

If you're running NixOs, I set up a flake with all the necessary packages, as well as cxx lib paths necessary for some of the pip packages:

```
nix develop --command bash
git lfs install
```

### Datasets, Model, and Packages

Use your favorite virtual environment and let's install the pip packages:

```
pip install -r requirements.txt
```

Then to install the preprocessed model and datasets, run

```
git lfs pull
```

Make sure to run this in the root directory of the entire repo. This should install a directory called `export/` containing `.pth` files.


## 📠 Preprocessing

If you want to train the model on a new folder of audio files, you can run

```
python preprocess.py --input path/to/audiofiles --output path/to/data.pth
```

The `audiofiles` directory expects a folder with the following form:

```
audiofiles/
 1-bass.mp3
 1-drums.mp3
 1-other.mp3
 1-vocal.mp3
 2-bass.mp3
 2-drums.mp3
 2-other.mp3
 2-vocal.mp3
 3-bass.mp3
 3-drums.mp3
 3-other.mp3
 3-vocal.mp3
...
```


## 🌒 Training and Inference

Simply run 

```
python train.py --data path/to/data.pth
```

to train the model on the features saves in `data.pth`. This should export a file called `model.pth` once training is finished (hoping to add checkpoints in the future). You can then run inference on another dataset with the inference script:

```
python inference.py ---model path/to/model.pth --data path/to/test-data.pth --plot
```

Using the `--plot` flag will generate some plots to evaluate accuracy.

