# Prediction_spam

## Description

This project was developed as part of my learning journey in Data Science, with a focus on supervised machine learning techniques applied to text classification. Using Python and libraries such as Scikit-learn, Pandas, and Matplotlib, I built a natural language processing pipeline to automatically identify SMS messages as either spam or non-spam (ham). The final model, based on an MLP neural network and text vectorization with CountVectorizer, achieved high accuracy and recall. The goal is to demonstrate the applicability of machine learning techniques in message filtering and in detecting textual patterns associated with unwanted content.

## Objective
The goal of this project is to apply supervised machine learning techniques to automatically classify text messages as either spam or non-spam (ham), based on their content. By building a natural language processing pipeline, the project aims to identify textual patterns associated with unwanted messages, contributing to the improvement of automatic filters in communication systems. The solution seeks to reduce the number of promotional messages reaching end users, enhancing both user experience and security in digital environments.

# Solution Planning

## Final Product

The final product of this project is an SMS message classification model capable of automatically distinguishing spam content from legitimate messages (ham). Using a machine learning pipeline with text vectorization via CountVectorizer and an MLP neural network classifier, the solution achieves high accuracy and recall in spam detection. The model can be easily integrated into message filtering systems, providing an efficient tool to reduce unwanted communications and enhance security and productivity in digital environments.

# Tools

Python 3.12.x: The project's primary programming language, widely adopted in data science for its simple syntax and robust ecosystem.

Pandas: Used for reading, manipulating, and analyzing tabular data from the CSV file.

NumPy: A fundamental library for mathematical operations and array manipulation.

Matplotlib: Used for basic graphical visualization during exploratory data analysis.

Scikit-learn: The main library for implementing the machine learning pipeline. It includes classes for text vectorization, data splitting, model construction, and performance evaluation.

CountVectorizer: Transforms message text into numerical representations (bag-of-words).

MLPClassifier: A multilayer neural network used as the classification model.

train_test_split: Splits the dataset into training, validation, and test sets.

classification_report: Evaluates metrics such as precision, recall, f1-score, and accuracy.

# Development
## Solution Strategy

To solve the problem of classifying messages as spam or non-spam, I used Python along with libraries such as Scikit-learn and Pandas. The process involved exploratory data analysis, transforming labels into binary values, and vectorizing the messages using CountVectorizer. The dataset was split into training, validation, and test sets, and a pipeline was built combining text vectorization with the MLPClassifier. The model was evaluated using metrics such as accuracy, recall, and F1-score, with a focus on effectively detecting spam messages and minimizing false negatives.

## The step-by-step

Step 1: Definition of the objective and scope of the analysis, focusing on the automatic detection of messages classified as spam or ham (non-spam) using supervised machine learning.

Step 2: Setting up the Python environment and installing the necessary libraries, such as Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

Step 3: Loading and initial exploration of the dataset, checking the structure, variable types, class distribution, and absence of null values.

Step 4: Creation of a new binary variable “Spam”, converting the categorical labels from the “Category” column into numerical values: 1 for spam and 0 for ham.

Step 5: Splitting the dataset into three subsets: training (60%), validation (20%), and testing (20%) using the train_test_split function from Scikit-learn.

Step 6: Vectorization of text messages with CountVectorizer, converting them into a sparse matrix based on word frequency (bag-of-words).

Step 7: Building a pipeline with Scikit-learn integrating the vectorization and the MLPClassifier, using one hidden layer with 100 neurons and a maximum of 500 iterations.

Step 8: Training the pipeline with the training data, followed by evaluation on the validation set using metrics such as accuracy, precision, recall, and F1-score.

Step 9: Final evaluation of the model on the test set, analyzing performance in spam message detection with a focus on the recall metric to reduce false negatives.

Step 10: Performing practical tests with simulated messages, identifying limitations such as class imbalance, and suggesting improvements such as using TF-IDF, balancing techniques, and hyperparameter tuning.

#  Model Training 

## Choice of the MLPClassifier Model

To classify the SMS messages in the dataset into two categories — spam and ham (non-spam) — the MLPClassifier (Multi-Layer Perceptron) algorithm was selected. This is a supervised neural network model from the Scikit-learn library. The choice of this classifier was based on technical criteria, performance, and alignment with the project's objectives, as detailed below:

Ability to Learn Complex Patterns
The MLPClassifier is a feedforward neural network capable of capturing non-linear patterns between the input data (vectorized text) and the output labels (spam/ham). Unlike linear models such as Naive Bayes, the neural network can learn more complex interactions between words and term combinations, which is particularly useful for detecting spam messages that are often written creatively to bypass simple filters.

Integration with Pipelines and Preprocessing
The choice of MLPClassifier was facilitated by its compatibility with Scikit-learn’s Pipeline, allowing text vectorization (CountVectorizer) and model training to occur in a seamless and modular flow. This makes the code cleaner, more reproducible, and easier to adapt for further validation or deployment steps.

Superior Performance in Practical Tests
During testing, the MLPClassifier achieved a high accuracy of 98.9% and a 92% recall for the spam class, exceeding expectations in detecting unwanted messages. The strong recall performance was particularly important, as the project prioritized minimizing false negatives — i.e., spam messages that are incorrectly classified as legitimate.

Efficiency and Complexity Control
Despite being a neural network, the chosen MLPClassifier was configured with just one hidden layer containing 100 neurons and a maximum of 500 iterations. This provided a good balance between performance and computational cost. This simplicity was well-suited to the dataset size (5,572 entries), helping to prevent overfitting and reduce training time.

Conclusion of the Choice
The MLPClassifier proved to be an effective and appropriate choice for the spam classification task in this project. Its ability to capture non-linear relationships, combined with easy pipeline integration and strong observed performance, supports its adoption as the primary model. While alternatives such as Naive Bayes, SVM, or deep neural networks exist, the MLP provided a robust, efficient, and interpretable solution aligned with the project goals and computational constraints.

# Results and Evaluation Metrics

The model’s performance was evaluated based on the validation and test datasets using classical supervised learning metrics: accuracy, precision, recall, and F1-score. The analysis placed special emphasis on the spam class, given the importance of minimizing false negatives—spam messages that are incorrectly classified as legitimate.

Performance on the Validation Set
During the validation phase, the model achieved impressive results:

Accuracy: 98.3%

Recall (spam): 88%

F1-score (spam): 93%

These results indicate that the model already demonstrated good generalization ability prior to the final testing phase, especially in correctly identifying spam messages, even in a class-imbalanced scenario (86.6% ham and 13.4% spam).

Performance on the Test Set
When evaluated on entirely new data, the model maintained a high level of performance:

Accuracy: 98.9%

Recall (spam): 92%

F1-score (spam): 96%

The 92% recall confirms that the model successfully identified the majority of spam messages, meeting the project's main objective. The 96% F1-score for the spam class reflects an excellent balance between precision and recall, ensuring that few legitimate messages were misclassified as spam (low false positive rate), and that most spam messages were correctly detected.


## Classification Report (Test):

![Classification_Report](
img/teste.png)

## Conclusions on the Results
The model demonstrated robustness, high precision, and excellent predictive capability, especially in identifying spam messages. Although some false negatives still occur (as shown in practical tests), the achieved results are fully satisfactory for real-world applications. This reinforces the effectiveness of the simple and focused pipeline used, while also leaving room for future improvements with class balancing, advanced vectorization techniques, and hyperparameter tuning.

# Conclusion
The SMS message classification project demonstrated the effectiveness of a simple yet well-structured pipeline based on CountVectorizer and MLPClassifier. The model achieved high levels of accuracy and recall, performing especially well in identifying legitimate messages (ham), with satisfactory results in detecting spam. However, the presence of some false negatives suggests room for improvement through more advanced techniques such as class balancing, more sophisticated text preprocessing, or the use of TF-IDF vectors. The developed solution proves promising for real-world applications in automatic message filtering systems, reinforcing the importance of steps such as data preparation, appropriate metric selection, and careful model validation in supervised machine learning problems.


