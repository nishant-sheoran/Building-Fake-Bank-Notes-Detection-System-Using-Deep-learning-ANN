# **Building Fake Bank Notes Detection System Using Deep Learning (ANN)**

## 📌 **Overview**  
Counterfeit banknotes pose a significant threat to the financial system, making it crucial to develop efficient detection mechanisms. This project leverages **Artificial Neural Networks (ANNs)** to classify banknotes as **genuine or fake** based on extracted image features. The model is trained on a dataset containing wavelet-transformed statistical attributes of banknote images.  

## 🎯 **Objective**  
The primary goal of this project is to build an **accurate and reliable classification model** that assists banks, financial institutions, and ATMs in detecting counterfeit currency, thereby enhancing security and preventing financial fraud.  

## 💂️ **Dataset Details**  
The dataset used in this project was sourced from **Kaggle**:  
🔗 [Banknote Authentication Dataset](https://www.kaggle.com/datasets/gauravduttakiit/banknote/data?select=train.csv)  

### **Features in the Dataset**  
- **VWTI (Variance of Wavelet Transformed Image)**  
- **SWTI (Skewness of Wavelet Transformed Image)**  
- **CWTI (Curtosis of Wavelet Transformed Image)**  
- **EI (Entropy of Image)**  
- **Class (Label: 1 for Genuine, 0 for Fake)**  

## 🛠️ **Project Workflow**  
The implementation follows a structured approach:  

1. **Data Preprocessing**  
   - Load and inspect the dataset.  
   - Perform feature scaling using `StandardScaler`.  
   - Split data into training and testing sets.  

2. **Model Development**  
   - Build an **Artificial Neural Network (ANN)** using TensorFlow/Keras.  
   - Train the model using the training dataset.  
   - Evaluate performance using **precision, recall, f1-score, and accuracy**.  

3. **Prediction System**  
   - Implement a **function for real-time prediction**.  
   - Preprocess input data and apply feature scaling.  
   - Use the trained model to classify banknotes as **Real or Fake**.  

## 📊 **Model Performance**  
The trained model achieves **high accuracy**, demonstrating its effectiveness in distinguishing between genuine and forged banknotes.  
Below is a summary of the **classification report**:  

| Metric       | Class 0 (Fake) | Class 1 (Real) |  
|-------------|--------------|--------------|  
| Precision   | 1.00        | 0.99        |  
| Recall      | 0.99        | 1.00        |  
| F1-Score    | 1.00        | 0.99        |  
| Accuracy    | **1.00 (100%)** |  

## 🚀 **How to Use**  
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/<username>/Building-Fake-Bank-Notes-Detection-System-Using-Deep-learning-ANN.git
   cd Building-Fake-Bank-Notes-Detection-System-Using-Deep-learning-ANN
   ```  
2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```  
3. **Run the Model**  
   ```bash
   python main.py
   ```  
4. **Make Predictions**  
   ```python
   input_data = np.array([[1.5, 2.3, 3.4, 0.7]])  
   result = make_prediction(input_data)  
   print(result)  # Outputs: "Real" or "Fake"
   ```  

## 🎥 **Video Tutorial**  
This project has a **YouTube tutorial** explaining the step-by-step implementation:  
🔗 [Watch the Tutorial Here](<https://www.youtube.com/watch?v=C_ecmIRVSpc&t=16s>)  

## 🤝 **Contributors**  
This README was created by **[Nishant Sheoran](https://github.com/nishant-sheoran)** to enhance the documentation for better accessibility and understanding. If you find this project helpful, consider ⭐ starring the repository!  

## 🔥 **Open Call for Contributions**  
We welcome contributions from the community! If you’d like to enhance this project, feel free to:  

- **Report Issues:** If you find a bug or have a feature request, open an issue.  
- **Submit Pull Requests:** Improve code, add documentation, or optimize performance.  
- **Enhance the Model:** Experiment with different architectures and improve accuracy.  

### **Steps to Contribute**  
1. **Fork the Repository**  
2. **Create a Feature Branch**  
   ```bash
   git checkout -b feature-branch
   ```  
3. **Make Changes and Commit**  
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```  
4. **Push to Your Fork and Create a Pull Request**  
   ```bash
   git push origin feature-branch
   ```  
5. **Submit a PR** and wait for review! 🎉  

We appreciate every contribution that helps improve this project! 🚀

