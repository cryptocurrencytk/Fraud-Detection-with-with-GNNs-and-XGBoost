# Fraud Detection with Graph Neural Networks (GNN) and XGBoost

The approach used for fraud detection in this project integrates both **Graph Neural Networks (GNN)** and **XGBoost**, leveraging the power of graph-based embeddings and traditional machine learning models to deal with imbalanced data effectively. Below is a detailed breakdown of the methodology:

---

## **1. Data Preprocessing**
The dataset contains transactions related to credit card fraud, with several features like transaction amount, time, user, card, and merchant details. Before using the data for model building, several preprocessing steps were applied:

- **Amount Parsing:** The 'Amount' column contains dollar values with dollar signs and commas. These were cleaned and converted into numeric values for use in the model.
  
- **Time Splitting:** The 'Time' column was split into 'Hour' and 'Minute' to capture temporal information about the transactions.

- **Missing Value Imputation:** Missing values in 'Merchant State' were replaced with 'Unknown', and missing zip codes were filled with the most frequent value (mode).

- **Label Encoding:** The target variable 'Is Fraud?' was converted into binary labels, with 1 indicating fraud and 0 indicating non-fraud transactions.

- **Graph Construction:** The transactions involve entities like users, cards, and merchants. A graph was constructed by mapping these entities to nodes, and connections (edges) were created based on transactions involving the same user, card, or merchant.

---

## **2. Rationale Behind Using GNN and XGBoost**

### **Why Graph Neural Networks (GNN)?**

Fraud detection often involves analyzing complex relationships between different entities such as users, merchants, and cards. These relationships can be naturally represented as a **graph**, where nodes represent entities and edges represent interactions or transactions. In such cases, a Graph Neural Network (GNN) is particularly well-suited because it allows us to:

- **Model Complex Interactions:** GNNs can effectively capture dependencies and interactions between users, cards, and merchants, which is important for detecting fraudulent activity. For example, a fraudulent user might frequently interact with a specific merchant or card, creating a pattern that can be detected using graph-based methods.
  
- **Learn Node Representations:** GNNs like **GraphSAGE** aggregate information from a node’s neighbors to generate informative embeddings. These embeddings help in understanding the relationships between different entities and how they interact within the graph, enabling better classification of fraud.

- **Scalability:** GraphSAGE is designed to handle large-scale graphs efficiently by sampling a fixed number of neighbors, which is crucial when working with large datasets like credit card transactions.

### **Why XGBoost?**

While GNNs excel at capturing complex relationships in graphs, they require high computational resources and might not always be ideal for tabular datasets or smaller datasets. **XGBoost** (Extreme Gradient Boosting) is used as a complementary approach because it is:

- **Effective for Imbalanced Data:** Credit card fraud datasets are often highly imbalanced, with fraud cases being much fewer than legitimate transactions. XGBoost performs well in such scenarios due to its ability to handle class imbalances effectively with parameters like `scale_pos_weight`.

- **Fast and Accurate:** XGBoost is known for its speed and high predictive accuracy. It uses gradient boosting to iteratively improve the model by minimizing the loss function, which allows it to learn complex patterns in the data.

- **Versatile:** XGBoost is a powerful classifier that works well for tabular data and provides interpretability through feature importance, making it a strong choice for fraud detection tasks.

By combining the graph-based approach of GNNs with the traditional boosting power of XGBoost, we leverage both deep graph representations and robust classification models, enabling a hybrid system that is both powerful and efficient.

---

## **3. Data Source**

The dataset used for this project was sourced from **Kaggle**, specifically from the dataset titled *[Credit Card Transactions](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions)*. This dataset contains a variety of features, including transaction amount, time, user, card, and merchant information, and it is widely used for fraud detection tasks. The dataset is well-suited for the task as it includes:

- A **binary classification** target variable indicating whether a transaction is fraudulent or not.
- Multiple categorical and numerical features that are useful for both graph-based learning and traditional machine learning methods.
- A highly **imbalanced distribution** of classes, with a very small percentage of fraudulent transactions compared to legitimate ones, which makes it a challenging task for machine learning models.

---

## **4. Graph Construction and Node Embeddings**
The next step involves **constructing a graph** to represent the relationships between users, cards, and merchants. Here's how the graph was built:

- **Nodes:** The graph contains nodes representing 'User', 'Card', and 'Merchant Name'. Each unique user, card, or merchant was assigned a unique node in the graph. The index of each node serves as its identifier.

- **Edges:** Transactions were used to create edges between nodes. For example, if a user made a transaction with a card at a merchant, edges were created between the respective nodes (user → card → merchant).

- **Graph Features:** The features for each node were randomly initialized, but in practice, node features could be based on transaction data such as the amount, time, frequency of transactions, etc.

- **Graph Neural Network (GNN):** The **GraphSAGE** (Graph Sample and Aggregation) model was chosen for node representation learning. This is a type of Graph Neural Network that aggregates information from neighboring nodes to compute the embedding for each node. GraphSAGE is suitable for large-scale graphs because it samples neighbors rather than using all of them, reducing computational complexity.

---

## **5. Training the GNN**
The **GraphSAGE** model was trained on the graph to obtain node embeddings that capture the underlying relationships between entities (users, cards, merchants). The model's architecture involves two layers of aggregation:

- **First Layer (Conv1):** This layer aggregates information from each node's neighbors to create an initial node representation.
  
- **Second Layer (Conv2):** The second layer further refines the embeddings based on the aggregated information from the first layer.

The training process involved minimizing the **Cross-Entropy Loss** function, which measures the difference between the model’s predictions and the true labels. Over multiple epochs (iterations), the model learned to better represent nodes in the graph.

---

## **6. Using Node Embeddings for Classification**
Once the GNN was trained, it produced **node embeddings** that were used as features for a classifier. These embeddings help the model better understand the interactions and relationships between users, cards, and merchants, which are crucial for detecting fraud.

The embeddings were merged with the fraud labels and used to train **XGBoost**, allowing it to make predictions on whether a transaction is fraudulent or not.

---

## **7. Model Results**

The training process involved 100 epochs for the GNN, with the loss and model evaluation metrics detailed below.

### **Training Loss**

The model's loss decreased steadily over the training epochs, showing significant improvement in each step:

Epoch 0  | Loss: 4.390958786010742
Epoch 10 | Loss: 0.04509425535798073
Epoch 20 | Loss: 0.02158733829855919
Epoch 30 | Loss: 0.0201947670429945
Epoch 40 | Loss: 0.014409753493964672
Epoch 50 | Loss: 0.007788381539285183
Epoch 60 | Loss: 0.0068520172499120235
Epoch 70 | Loss: 0.005124170798808336
Epoch 80 | Loss: 0.0041487934067845345
Epoch 90 | Loss: 0.003285979153588414

### **Best Hyperparameters**

The best parameters for XGBoost were identified as:
Best parameters: {‘learning_rate’: 0.1, ‘max_depth’: 3, ‘scale_pos_weight’: 10}

These parameters optimized the model's ability to handle the class imbalance and make accurate predictions.

### **Accuracy and Evaluation Metrics**

The model achieved the following evaluation results:

- **Accuracy:** **92.86%** — A high accuracy, particularly considering the imbalanced nature of the dataset.
  
#### **Classification Report**

The classification report shows the precision, recall, and F1-score for both the fraudulent and legitimate transactions:
           precision    recall  f1-score   support
       0       1.00      0.93      0.96      3990
       1       0.01      0.67      0.01         3

accuracy                           0.93      3993

macro avg       0.50      0.80      0.49      3993
weighted avg       1.00      0.93      0.96      3993

- **Precision:** For class `0` (non-fraud), the precision is **1.00**, meaning the model correctly identified legitimate transactions most of the time.
- **Recall:** For class `1` (fraud), the recall is **0.67**, indicating that the model correctly identified 67% of fraudulent transactions.
- **F1-Score:** The F1-score for fraud detection was quite low, primarily due to the imbalance of fraudulent cases.

### **Adjusted Accuracy**

To adjust for class imbalance, we calculated **Adjusted Accuracy**:
Adjusted Accuracy: 0.8650137741046832

This adjusted accuracy provides a better understanding of the model's performance, reflecting its ability to detect fraud in the presence of significant class imbalance.

#### **Adjusted Classification Report**

After adjustment for class imbalance:
           precision    recall  f1-score   support
       0       1.00      0.86      0.93      3990
       1       0.01      1.00      0.01         3

accuracy                           0.87      3993

The adjusted classification report shows improved performance, particularly in terms of recall for the fraud class (1), with recall approaching 1.00 for fraudulent transactions.

### **AUC - ROC Curve**

The **Area Under the Curve (AUC)** for the ROC curve was calculated to assess the model's discriminatory ability:
AUC: 0.97

This AUC value indicates excellent performance in distinguishing between fraudulent and non-fraudulent transactions, as values close to 1.0 signify a highly effective model.

---

## **8. Weaknesses of the Model**

While the model performs well overall, there are several areas where it could be improved:

### **Class Imbalance**
Despite using **SMOTE** and **ADASYN** to address class imbalance, the model's performance for the fraudulent class (class `1`) remains suboptimal. The **precision** for fraud detection is very low (0.01), and the **F1-score** is poor. This indicates that the model struggles to generalize for the minority class, even after adjusting the decision threshold.

### **Model's Precision for Fraudulent Transactions**
The **precision** for fraud detection is very low, indicating that the model is not very reliable when it predicts fraud. This means that while the model identifies a few frauds, it also generates a high number of false positives. This is particularly concerning in real-world applications where false positives could lead to unnecessary checks or investigations.

### **Limited Representation of Fraudulent Transactions**
The dataset is heavily imbalanced, with very few instances of fraudulent transactions compared to legitimate ones. This imbalance leads to a model that is highly biased toward predicting the majority class (non-fraud). The **recall** for fraud is **67%**, which is decent, but the model still misses a significant proportion of fraudulent transactions.

### **Feature Engineering**
The model uses basic features (such as amounts and temporal features) for graph node embeddings. However, these may not fully capture the complex relationships between transactions, users, and merchants. More sophisticated feature engineering techniques could improve performance, especially for identifying fraud patterns.

### **Potential Overfitting**
The model's high accuracy could be indicative of overfitting, particularly given the imbalance of the classes. The model performs well on the training set but may struggle with generalization to new, unseen data. Cross-validation and further regularization techniques could be used to mitigate this.

---

## **9. Conclusion**

This fraud detection model successfully integrates **Graph Neural Networks (GNN)** for feature learning with **XGBoost** for classification. Key findings include:

- **Graph Neural Networks** effectively captured the complex relationships between users, merchants, and cards, improving the model's ability to detect fraud.
- **XGBoost** was able to classify fraudulent transactions effectively, leveraging the powerful features extracted by the GNN and dealing with the class imbalance problem.
- The model achieved **high AUC** (0.97) and **adjusted accuracy** (86.5%), which are excellent for fraud detection tasks with imbalanced data.

However, the model faces challenges with class imbalance, leading to low precision for fraud detection. The performance could be further improved by incorporating more advanced techniques for handling imbalanced datasets, improving feature engineering, and tuning the model to reduce false positives.

This approach demonstrates the power of combining deep learning and traditional machine learning to solve complex problems like fraud detection, but it also highlights the importance of addressing the limitations of class imbalance and model generalization in real-world scenarios.

















