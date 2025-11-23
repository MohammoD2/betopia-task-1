# 🔥 Advanced ICP Matching Engine

An intelligent **Ideal Customer Profile (ICP) Matching Engine** that uses AI-powered semantic similarity to find and rank companies that best match your target customer profile. This tool helps businesses identify potential customers by analyzing company descriptions, industries, and tech stacks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project provides two ways to match companies against your ICP:

1. **Streamlit Web Application** (`App.py`) - Interactive web interface for easy use
2. **Standalone Python Script** (`ICP Matching Engine..py`) - Command-line version for automation

The engine uses advanced **sentence transformers** and **Maximal Marginal Relevance (MMR)** algorithms to:
- Calculate semantic similarity between your ICP description and company profiles
- Rank companies by relevance score
- Generate intelligent explanations for why each company matches your ICP

## ✨ Features

- 🤖 **AI-Powered Matching**: Uses state-of-the-art sentence transformer models (BAAI/bge-base-en-v1.5)
- 📊 **Semantic Similarity**: Understands context and meaning, not just keywords
- 🎯 **MMR Explanations**: Generates diverse, relevant explanations using Maximal Marginal Relevance
- 📈 **Ranked Results**: Companies sorted by similarity score (0-1 scale)
- 💾 **Export Results**: Download results as JSON for further analysis
- 🎨 **User-Friendly UI**: Clean Streamlit interface for non-technical users
- ⚙️ **Configurable**: Customize embedding models and explanation parameters

## 🔧 How It Works

1. **Data Loading**: Reads company data from CSV file (130 companies included)
2. **Feature Combination**: Combines company description, industry, tech stack, and location
3. **Embedding Generation**: Converts text into numerical vectors using sentence transformers
4. **Similarity Calculation**: Computes cosine similarity between ICP and company embeddings
5. **Ranking**: Sorts companies by similarity score (highest first)
6. **Explanation Generation**: Uses MMR to select the most relevant and diverse sentences explaining the match

### Technical Details

- **Embedding Model**: BAAI/bge-base-en-v1.5 (or bge-small-en-v1.5 for faster processing)
- **Similarity Metric**: Cosine similarity on normalized embeddings
- **MMR Algorithm**: Balances relevance (60%) and diversity (40%) for explanations

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Setup

1. **Clone or download this repository**

2. **Navigate to the project directory**
   ```bash
   cd Task-1
   ```

3. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment**
   
   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `streamlit` - Web application framework
   - `pandas` - Data manipulation
   - `torch` - PyTorch for deep learning
   - `sentence-transformers` - Pre-trained embedding models
   - `numpy` - Numerical computing
   - `scikit-learn` - Machine learning utilities

6. **Verify installation**
   ```bash
   python --version
   pip list
   ```

## 📖 Usage

### Option 1: Streamlit Web Application (Recommended)

1. **Start the Streamlit app**
   ```bash
   streamlit run App.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

3. **Configure your ICP** (in the sidebar):
   - **Select Embedding Model**: Choose between `bge-base-en-v1.5` (more accurate) or `bge-small-en-v1.5` (faster)
   - **Enter ICP Description**: Describe your ideal customer profile
     - Example: *"We are looking for e-commerce and fashion companies in Bangladesh with scalable online presence."*
   - **Set Explanation Sentences**: Number of sentences to include in explanations (1-10, default: 4)

4. **Click "🚀 PROCESS"** button

5. **View Results**:
   - Top 10 matching companies displayed in a table
   - Download full results as JSON using the download button

### Option 2: Standalone Python Script

1. **Edit the script** (optional):
   - Open `ICP Matching Engine..py`
   - Modify `ICP_DESCRIPTION` variable (line 92-95) to match your needs
   - Adjust `EMBEDDING_MODEL` if needed (line 12)

2. **Run the script**
   ```bash
   python "ICP Matching Engine..py"
   ```

3. **Check the output**:
   - Results saved to `icp_ranked_companies_advanced.json`
   - Top 10 preview printed in terminal

### Example ICP Descriptions

Here are some example ICP descriptions you can use:

```
# E-commerce companies
"We are looking for e-commerce and fashion companies in Bangladesh with scalable online presence."

# Tech startups
"Early-stage SaaS companies in fintech with 50-200 employees, using modern cloud technologies."

# Gaming companies
"Mobile game developers using Unity, focused on hyper-casual games with strong user retention."

# Blockchain companies
"Web3 and blockchain gaming companies building play-to-earn economies with NFT integration."
```

## 📁 Project Structure

```
Task-1/
│
├── App.py                              # Streamlit web application
├── ICP Matching Engine..py            # Standalone Python script
├── ICP Matching Engine..ipynb         # Jupyter notebook version
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
│
├── Data/
│   └── Task-1-data(130 company).csv  # Company dataset (130 companies)
│
├── venv/                              # Virtual environment (created during setup)
│
└── icp_ranked_companies_advanced.json # Output file (generated after running)
```

## ⚙️ Configuration

### Embedding Models

Two pre-trained models are available:

- **`BAAI/bge-base-en-v1.5`** (Default)
  - More accurate results
  - Slower processing (~2-3 minutes for 130 companies)
  - Recommended for production use

- **`BAAI/bge-small-en-v1.5`**
  - Faster processing (~30-60 seconds)
  - Slightly less accurate
  - Good for testing and quick iterations

### MMR Parameters

The MMR (Maximal Marginal Relevance) algorithm uses these weights:
- **Relevance weight**: 0.6 (60%)
- **Diversity weight**: 0.4 (40%)

These can be modified in the `generate_mmr_explanation()` function (line 64 in both files).

## 📊 Output Format

The output JSON file contains an array of company objects:

```json
[
  {
    "company_name": "Company Name",
    "similarity_score": 0.8542,
    "explanation": "This company matches the ICP because: [MMR-selected sentences]"
  },
  ...
]
```

**Fields:**
- `company_name`: Name of the company
- `similarity_score`: Cosine similarity score (0.0 to 1.0, higher is better)
- `explanation`: AI-generated explanation of why the company matches

## 📦 Requirements

See `requirements.txt` for the complete list. Main dependencies:

- **streamlit** >= 1.0.0
- **pandas** >= 1.0.0
- **torch** >= 1.9.0
- **sentence-transformers** >= 2.0.0
- **numpy** >= 1.20.0
- **scikit-learn** >= 0.24.0

## 🔍 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```
Solution: Make sure you've activated your virtual environment and installed requirements:
  pip install -r requirements.txt
```

**2. CUDA/GPU Issues**
```
Solution: The code works on CPU by default. If you have CUDA errors, ensure PyTorch is installed correctly:
  pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**3. File Not Found Error**
```
Solution: Ensure the CSV file exists at: Data/Task-1-data(130 company).csv
  Check that the file path is correct and the Data folder exists.
```

**4. Slow Processing**
```
Solution: 
  - Use the smaller model: bge-small-en-v1.5
  - Reduce the number of companies in your dataset
  - Ensure you have sufficient RAM (recommended: 8GB+)
```

**5. Streamlit Not Opening**
```
Solution: 
  - Check if port 8501 is already in use
  - Try: streamlit run App.py --server.port 8502
  - Manually open http://localhost:8501 in your browser
```

### Performance Tips

- **First Run**: The embedding model downloads automatically (~400MB for base model). This happens only once.
- **Batch Processing**: The script processes companies in batches of 32 for efficiency.
- **Memory Usage**: Processing 130 companies requires approximately 2-4GB RAM.

## 📝 Notes

- The first time you run the application, the embedding model will be downloaded (this may take a few minutes depending on your internet speed).
- Results are deterministic - the same ICP description will produce the same rankings.
- The similarity scores are normalized between 0 and 1, where 1.0 indicates a perfect match.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is provided as-is for educational and business use.

---

**Need Help?** Check the troubleshooting section above or review the code comments for detailed explanations of each function.

