# Auto-Insurance-Fradulent-Claims-ML-Detection-
Three different ML classification models detecting fradulent claims aimed at insurance companies

## Dataset

The dataset used in this project is the **Auto Insurance Claims Data** from Kaggle.

### Download Instructions

1. Visit the dataset page: [Auto Insurance Claims Data](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)

2. **Option A: Manual Download**
   - Click the "Download" button on the Kaggle page
   - Extract the downloaded zip file
   - Place `insurance_claims.csv` in the `data/raw/` directory

3. **Option B: Using Kaggle CLI**
   ```bash
   # Install Kaggle CLI (if not already installed)
   pip install kaggle
   
   # Set up your Kaggle API credentials
   # Download your kaggle.json from https://www.kaggle.com/settings
   # Place it in ~/.kaggle/kaggle.json
   
   # Download the dataset
   kaggle datasets download -d buntyshah/auto-insurance-claims-data -p data/raw/ --unzip
   ```
