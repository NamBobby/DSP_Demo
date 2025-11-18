# EEG Sleep Stage Classification  
### *FPT School of Business & Technology (FSB)*  
### **Group 3 – DSP501 Final Project**

This repository contains the demo implementation of our Digital Signal Processing (DSP) + Machine Learning (ML) pipeline for **automatic EEG sleep stage classification**.  
The application includes:

- Full DSP preprocessing pipeline  
- Feature extraction (statistical, Hjorth, PSD)  
- Logistic Regression / SVM / Random Forest models  
- Streamlit app for real-time prediction  
- Public reproducible code & report  

---

# Project Resources

| Resource | Link |
|---------|------|
| **GitHub Demo Repository** (main implementation) | https://github.com/NamBobby/DSP_Demo |
| **Kaggle Notebook** (core contribution) | https://www.kaggle.com/code/namle25/eeg-sleep-stage-dsp-implementation |
| **Overleaf Technical Report** | https://www.overleaf.com/read/vjnzpkvsxbfs#2594d5 |

---


# 1. Clone or Download the Project

If using GitHub:

```bash
git clone https://github.com/your-repo/eeg_sleep_demo.git
cd eeg_sleep_demo
```

If downloaded as `.zip`, extract and move into the project folder:

```bash
cd eeg_sleep_demo
```

---

# 2. Create a Virtual Environment (.venv)

This ensures a clean and isolated Python environment.

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

You should now see something like:

```
(.venv) user@computer:~/eeg_sleep_demo
```

---

# 3. Install Dependencies

If your project already has a `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you want to manually install the required libraries:

```bash
pip install streamlit numpy pandas scikit-learn seaborn matplotlib joblib
```

---

# 4. (Optional) Generate a Fresh requirements.txt

If you've installed libraries and want to export the exact environment:

```bash
pip freeze > requirements.txt
```

This ensures others can reproduce the same environment.

---

# 5. Verify Model & Resource Files

Make sure the following files exist before running the app:

* `models/logistic_regression_base.pkl`
* `models/logistic_regression_ica.pkl`
* `results/label_map.json`
* `data_demo/demo_sample.csv`
* `app.py`

If any file is missing → the app will fail to load.

---

# 6. Run the Streamlit Application

Inside the activated `.venv`, run:

```bash
streamlit run app.py
```

You will see output like:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

Open the link to access the UI.

---

# 7. How to Use the App

Inside the Streamlit interface, you can:

### Select Model:

* **BASE model** (trained without ICA)
* **ICA model** (artifact-cleaned features)

### Choose Input Type:

* Random feature vector (for testing UI)
* Manual input (24 comma-separated values)
* Pre-loaded `demo_sample.csv`
* Upload your own `.csv` file (must have exactly **24 numeric values**)

### The App Will Display:

* Predicted sleep stage label (Wake / N1 / N2 / N3 / REM)
* Class ID (integer)
* Probability bar chart
* A final summary explaining the result

---

# 8. Format of External Input File

Your `.csv` must contain:

* Exactly **one row**
* Exactly **24 values** (the DSP features for one 30-second epoch)
* No header needed

Example (`data_demo/demo_sample.csv`):

```
0.15, -0.23, 0.88, -1.23, 0.55, 
1.14, 0.22, -0.44, 0.01, 0.77,
1.33, 0.06, -0.92, 0.55, 0.29,
1.12, -0.42, 0.11, 0.21, -0.85,
0.74, 0.02, 1.34, -0.55
```

This file simulates an unseen external 30-second EEG epoch after DSP feature extraction.

---

# 9. Example Prediction Output

After uploading an input file, you might see:

```
Predicted Sleep Stage: Sleep stage 2
Class ID: 2
```

Followed by a bar chart showing the probability distribution.

This demonstrates the complete DSP → Feature Extraction → ML inference process.

---

# 10. Deactivate Virtual Environment

When you're done:

### macOS / Linux

```bash
deactivate
```

### Windows

```powershell
deactivate
```

