# previsao_spam
## Descrição
This project was developed as part of my learning journey in Data Science, with a focus on supervised machine learning techniques applied to text classification. Using Python and libraries such as Scikit-learn, Pandas, and Matplotlib, I built a natural language processing pipeline to automatically identify SMS messages as either spam or non-spam (ham). The final model, based on an MLP neural network and text vectorization with CountVectorizer, achieved high accuracy and recall. The goal is to demonstrate the applicability of machine learning techniques in message filtering and in detecting textual patterns associated with unwanted content.
## Objetivo
The goal of this project is to apply supervised machine learning techniques to automatically classify text messages as either spam or non-spam (ham), based on their content. By building a natural language processing pipeline, the project aims to identify textual patterns associated with unwanted messages, contributing to the improvement of automatic filters in communication systems. The solution seeks to reduce the number of promotional messages reaching end users, enhancing both user experience and security in digital environments.
# Planejamento da Solução
## Produto final
The final product of this project is an SMS message classification model capable of automatically distinguishing spam content from legitimate messages (ham). Using a machine learning pipeline with text vectorization via CountVectorizer and an MLP neural network classifier, the solution achieves high accuracy and recall in spam detection. The model can be easily integrated into message filtering systems, providing an efficient tool to reduce unwanted communications and enhance security and productivity in digital environments.
# Ferramentas
Python 3.12.x: The project's primary programming language, widely adopted in data science for its simple syntax and robust ecosystem.

Pandas: Used for reading, manipulating, and analyzing tabular data from the CSV file.

NumPy: A fundamental library for mathematical operations and array manipulation.

Matplotlib: Used for basic graphical visualization during exploratory data analysis.

Scikit-learn: The main library for implementing the machine learning pipeline. It includes classes for text vectorization, data splitting, model construction, and performance evaluation.

CountVectorizer: Transforms message text into numerical representations (bag-of-words).

MLPClassifier: A multilayer neural network used as the classification model.

train_test_split: Splits the dataset into training, validation, and test sets.

classification_report: Evaluates metrics such as precision, recall, f1-score, and accuracy.
# Desenvolvimento
## Estratégia da Solução
To solve the problem of classifying messages as spam or non-spam, I used Python along with libraries such as Scikit-learn and Pandas. The process involved exploratory data analysis, transforming labels into binary values, and vectorizing the messages using CountVectorizer. The dataset was split into training, validation, and test sets, and a pipeline was built combining text vectorization with the MLPClassifier. The model was evaluated using metrics such as accuracy, recall, and F1-score, with a focus on effectively detecting spam messages and minimizing false negatives.
## O passo a passo
