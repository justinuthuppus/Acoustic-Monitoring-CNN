# Acoustic Patient Monitoring CNN (ECEN 250 Project)

This repository contains the architecture, data pipeline, and performance results for a Convolutional Neural Network (CNN) built for an ECEN 250 engineering lab project. The system is designed as a non-invasive acoustic monitoring tool for nursing homes or assisted living facilities to automatically flag potentially dangerous patient events (like falls, standard appliance malfunctions, or medical distress) by classifying ambient environmental sounds.

> **⚠️ Academic Integrity & Privacy Notice:** To comply with university academic honesty guidelines, the raw Python notebooks (`.ipynb`) and model checkpoints (`.keras`) are hosted in a private repository. Full code access can be instantly granted to recruiters and engineering managers upon request—please reach out via email or LinkedIn!

---

## Project Overview & Context
In this project, my team was tasked with building a machine learning system that could assist healthcare workers in senior care settings. The goal was to build an automated audio classifier that could monitor ambient sounds and instantly flag anomalies, such as a patient slipping in the shower or an industrial machine failing. 

Because we were dealing with standard 1D audio waves that are difficult for basic linear models to classify directly, our approach was to transform the audio signals into visual data (spectrograms) and leverage Computer Vision techniques using a Convolutional Neural Network.

## Technical Workflow & Data Pipeline

### 1. Digital Signal Processing (DSP) & Audio Windowing
* **Audio Formatting:** Raw `.wav` audio signals were first captured, set to a standardized frame rate of 44,100Hz, converted to a single channel (mono), and amplitude-normalized using `pydub`.
* **Windowing:** Long audio streams were parsed into overlapping 1-second chunks (with a 0.5-second step overlap) to ensure transient acoustic events weren't split or lost between frames.
* **Feature Extraction:** Using `scipy.signal.spectrogram`, the 1D audio sample windows were converted into 2D Time-Frequency Spectrogram arrays, mapping the exact frequency changes over time into visual matrices.

### 2. CNN Architecture & Layout
The core deep learning model was built sequentially using TensorFlow and Keras. Since the system is intended to eventually run on low-power edge devices inside facilities, the network was designed to be deep but highly efficient:
* **Input Layer:** Processes the generated spectrogram plots reshaped to $256 \times 256 \times 3$ tensors.
* **Convolutional Layers:** Two 2D Convolution layers (`Conv2D`) utilizing 8 and 16 filters respectively with a $3 \times 3$ kernel size to extract local acoustic features.
* **Regularization & Pooling:** Intermittent `MaxPooling2D` layers (downsampling via $2 \times 2$ grids) to reduce spatial dimensions, paired with `Dropout` layers set at $25\%$ to mitigate model overfitting on the training set.
* **Dense Output Layer:** A fully connected `Flatten` and `Dense` network with a `softmax` activation function spitting out probabilities for the 8 distinct target sound classes.

### 3. Handling Dataset Imbalances
During initial testing, the baseline model suffered from majority-class training bias due to uneven data sizes across classes. To fix this without artificially inflating data, I implemented an inverse frequency class-weighting array (`class_weight.compute_class_weight`) from `sklearn` during model compilation. This penalized the categorical cross-entropy loss function more heavily when misclassifying rarer target sounds, forcing the network to balance its sensitivity evenly.

---

## Key Performance Results
* **Target Classes Classified:** `bottle_drop`, `brushing_teeth`, `chainsaw`, `drying_machine`, `glass_breaking`, `snoring`, `vacuum_cleaner`, `washing_machine`.
* **Peak Training Accuracy:** Achieved **95.72% accuracy** over 35 training epochs.
* **Loss Optimization:** Successfully minimized loss down to a categorical cross-entropy of **0.1224**.

---

## Tech Stack & Tools Used
* **Languages:** Python, Markdown
* **Deep Learning Frameworks:** TensorFlow, Keras
* **Data Science & Math Libraries:** NumPy, SciPy, Scikit-Learn
* **Audio Processing:** Pydub, FFmpeg, Wavfile
* **Version Control & Collaboration:** Git, GitHub, Microsoft Live Share
