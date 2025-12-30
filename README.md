# 🌧️ Rainfall Prediction App

## Setup & Deployment Instructions

### Step 1: Train the Models (Run FIRST)
Before running the Streamlit app, you need to train and save the models:

```bash
python train_model.py
```

**What this does:**
- Loads the `weatherAUS.csv` dataset
- Preprocesses and engineers features
- Selects the top 5 features using SelectKBest
- Trains a Random Forest Regression model
- Saves `RF_model.pkl` and `scaler.pkl` files

**Expected Output:**
```
🌧️ Starting Model Training...
📂 Loading dataset...
✓ Dataset loaded: (xxxx, xxx)
✓ Data preprocessing complete
...
✅ Training Complete! Models are ready for deployment.
```

### Step 2: Run Streamlit App
Once training is complete, launch the app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Features

✨ **Interactive Prediction Interface**
- Slide-based input for weather parameters
- Real-time rainfall predictions
- Rainfall intensity classification
- Model performance metrics

🎯 **Input Parameters:**
- Minimum Temperature (°C)
- Maximum Temperature (°C)
- Rainfall (mm)
- Evaporation (mm)
- Sunshine (hours)

📊 **Output:**
- Predicted rainfall amount (mm)
- Rainfall intensity level (No Rain / Light / Moderate / Heavy)
- Input summary

---

## Troubleshooting

**Error: "Models not found!"**
→ Run `python train_model.py` first

**Error: "weatherAUS.csv not found"**
→ Make sure the CSV file is in the same directory as the scripts

**Streamlit won't launch**
→ Install Streamlit: `pip install streamlit`

---

## Files Required

```
📁 Your Project Folder/
├── app.py                 (Streamlit app)
├── train_model.py         (Training script)
├── python.ipynb          (Jupyter notebook with analysis)
├── weatherAUS.csv        (Dataset)
├── RF_model.pkl          (Generated after training)
└── scaler.pkl            (Generated after training)
```

---

## Requirements

```
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
```

Install with: `pip install -r requirements.txt`
