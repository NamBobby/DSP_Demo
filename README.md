# EEG Sleep Stage Classification Demo (DSP + ML)

This project is a **demo application** for our DSP501 final project:  
**"EEG Sleep Stage Classification via Digital Signal Processing and Machine Learning"**.

The goal of this demo is to show that our trained model can:
1. Load a pre-trained sleep stage classifier (trained on Sleep-EDF using DSP features),
2. Accept an external 30s epoch **feature vector** as input (24 DSP features),
3. Predict the corresponding **sleep stage** (Wake, N1, N2, N3, REM),
4. Visualize the class probability distribution.

---

## Project Structure

```bash
eeg_sleep_demo/
├── app.py                    # Streamlit demo app
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/
│   ├── logistic_regression_base.pkl   # Model trained on BASE features
│   └── logistic_regression_ica.pkl    # Model trained on ICA-cleaned features
├── results/
│   ├── label_map.json                 # id2label + label2id mapping
│   └── best_models_summary.json       # (optional) best model info
└── data_demo/
    └── demo_sample.csv                # Example external input feature vector
