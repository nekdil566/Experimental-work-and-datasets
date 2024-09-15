# Sentiment and Emotion Analysis in User Reviews

## Overview

This repository contains the experimental work and datasets for analyzing sentiment and emotion in user reviews. The project explores various deep learning models, such as GRU, BiGRU,LSTM, and BiLSTM, CNN, RNN and BiRNN implements sentiment analysis. Additionally, the repository includes code snippets in PDF format, which users can copy and use directly from the documents.

## Directory Structure

### Code Files
- **Algorithm 1 Processing Reviews for Sentiment.ipynb**: Notebook focused on the initial processing of reviews to extract and analyze sentiment.
- **BIGRU_MODEL.ipynb**: Implements a BiGRU model to capture contextual sentiment in user reviews.
- **BILSTM_OVER_UNDER_SAMPLING.ipynb**: This notebook addresses class imbalance using over-sampling and under-sampling techniques, applied to a BiLSTM model.
- **GRU_model.ipynb**: Implements a GRU model for sequence processing tasks such as sentiment analysis.
- - **RR2_CNN_model.ipynb**: Implements a CNN model for sequence processing tasks such as sentiment analysis.
- **Grid search for(chatgptdataset).ipynb**: Performs hyperparameter optimization using grid search on models trained with the ChatGPT dataset.
- **LSTM_OVER_UNDER_LSTM.ipynb**: Focuses on LSTM models with sampling techniques to manage class distribution.
- **lstm_bilstm.ipynb**: Compares LSTM and BiLSTM models in terms of performance on sentiment analysis tasks.
- **Vadar setiment anylysis.ipynb**: Implements VADER sentiment analysis for text data.

### PDF Files (Code in PDF)
- **R2_BIRNN.pdf**: Contains detailed explanations or code snippets related to Bidirectional RNN models. Users can copy the code from this PDF.
- **R2_RNN_code.pdf**: Provides RNN code and explanations. Users are encouraged to copy and utilize the code in their own projects.
- **Understanding Emotions in End.pdf**: Explores theoretical aspects of understanding emotions in user reviews.

### Datasets
- **R2_ChatGPt_dataset.csv**: Dataset related to interactions with ChatGPT,annotated for sentiment or emotion.
- **chatgptannoated_datset_final.csv**: Finalized version of the annotated ChatGPT dataset.
- **end-user reviews dataset.csv**: Contains end-user reviews, which may include sentiment or emotion annotations.
- **vadar_resutls.csv**: Results from VADER sentiment analysis on the provided datasets.

## Requirements

To run the code provided in this repository, you will need the following:

- **Python 3.x**: The primary programming language used in this project.
- **Jupyter Notebook or JupyterLab**: For running and interacting with `.ipynb` files.
- **Python Libraries**:
  - `pandas`: For data manipulation and analysis.
  - `numpy`: For numerical computations.
  - `matplotlib` or `seaborn`: For data visualization.
  - `scikit-learn`: For machine learning and preprocessing.
  - `keras` and `tensorflow`: For building and training deep learning models.
  - `nltk` or `vaderSentiment`: For sentiment analysis using VADER.

## Running the Analysis

1. **Exploring the Datasets**: Start by loading the datasets provided (`R2_ChatGPt_dataset.csv`, `chatgptannoated_datset_final.csv`, `end-user reviews dataset.csv`) to understand their structure and content.
  
2. **Sentiment Analysis**:
   - Use the **Vadar setiment anylysis.ipynb** to perform sentiment analysis using VADER.
   - Explore deep learning models using the other notebooks (`GRU_model.ipynb`, `BIGRU_MODEL.ipynb`, etc.) to classify sentiment or emotions in the datasets.

3. **Code from PDFs**: For those interested in using code directly from the PDFs (`R2_BIRNN.pdf`, `R2_RNN_code.pdf`), simply copy the code snippets provided and adapt them to your datasets or research needs.

## Contact Information

For questions or feedback regarding this project, please contact [naikdil2003@gmail.com].


