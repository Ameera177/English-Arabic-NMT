
# English-Arabic Neural Machine Translation (NMT)

This project implements a Neural Machine Translation (NMT) system to translate English sentences into Arabic using TensorFlow and Keras. It employs a Sequence-to-Sequence (Seq2Seq) architecture with Bidirectional LSTM layers and an attention mechanism to improve translation accuracy. The dataset was downloaded from Kaggle and cleaned to remove noise, punctuation, and digits. Tokenization and padding were applied, followed by training the model. A simple Gradio interface is provided for testing translations interactively.

## 🛠️ Tools Used

- **TensorFlow / Keras** – For building and training the NMT model  
- **NumPy** – For numerical operations  
- **Pandas** – For data loading and cleaning  
- **re / string** – For text preprocessing  
- **Pickle** – To save/load tokenizers  
- **scikit-learn** – For splitting the dataset (train/test)  
- **Kaggle API** – To download the dataset  
- **Google Colab** – For model development and training  
- **Gradio** – For building an interactive web interface  
