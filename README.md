# Fake News Detection in Tweets

This project focuses on the segmentation of tweets to determine whether a tweet contains fake news or not. By training a model using SparkML on pre-labeled CSV data stored in Hadoop, we aim to classify incoming tweets in real-time. The project utilizes Apache Spark and Kafka for tweet ingestion and consumption, and MariaDB for serving the transformed data. The Kappa architecture is used for this project: ![Kappa Architecture](/Users/thomastruyts/Downloads/Results.png)


## Dataset Description

The dataset used for training the model consists of pre-labeled tweets stored in a CSV file format. The tweets are classified as either containing fake news or being legitimate. The dataset is stored in Hadoop, allowing for distributed processing with Spark. 

## Machine Learning Model

The machine learning model employed in this project is the Random Forest Classifier, trained using SparkML. Random Forest is a popular ensemble learning algorithm that combines multiple decision trees to make predictions. It is particularly suitable for classification tasks and can handle a large number of features.

## Feature Engineering Techniques

To preprocess the tweet data and extract meaningful features, several techniques are employed:

1. Tokenization: The tweets are split into individual tokens or words. This process helps to break down the text into smaller units, enabling further analysis.

2. CountVectorizer: This technique is used to convert a collection of text documents (tweets in this case) into a matrix of token counts. It captures the frequency of each word in the tweet, providing valuable information for classification.

3. TF-IDF (Term Frequency-Inverse Document Frequency): TF-IDF is a numerical statistic that reflects the importance of a word in a tweet. It combines term frequency (TF) and inverse document frequency (IDF) to assign weights to words. This technique helps to emphasize words that are more relevant to the classification task.

These feature engineering techniques help to transform the raw tweet data into a sparse matrix representation, which is suitable for training the machine learning model.

## Model Evaluation and Accuracy

After parameter tuning and training the Random Forest Classifier, the model achieved an accuracy of 70%. This accuracy metric indicates the proportion of correctly classified tweets compared to the total number of tweets. Further improvements in accuracy can be achieved by exploring other machine learning algorithms, such as Word2Vec, which can enhance the tweet processing capabilities.

## Serving and Data Storage

For serving the data and storing the transformed tweets, MariaDB is utilized. Since the project deals with structured small data, MariaDB offers a reliable and efficient solution. The transformed tweets are pushed into predefined tables, enabling easy retrieval and analysis.

## Further Improvements

To enhance the model's performance and capabilities, the following steps can be considered:

1. Word2Vec: Word2Vec is a technique for vectorizing words that captures semantic relationships. By incorporating Word2Vec as a tweet processor, the model can gain a deeper understanding of the textual content and improve classification accuracy.

2. Explore other ML Algorithms: Experimenting with different machine learning algorithms, such as Support Vector Machines (SVM) or Multilayer Perceptron (MLP), can provide insights into their performance on the given task and potentially lead to better predictions.

3. BI Platform Integration: Connecting a Business Intelligence (BI) platform to the MariaDB database can enable visualization, reporting, and further analysis of the tweet data. This integration can provide valuable insights and facilitate decision-making processes.

## Repository Contents

- `fake_news_detection.ipynb`: A Jupyter Notebook containing the Python code and step-by-step explanation for building the SparkML model and tweet segmentation.
- `dataset.csv`: The pre-labeled dataset used for training the model.
- `README.md`: This file, providing an overview of the project.

## Requirements

To run the Jupyter Notebook and reproduce the results, ensure the following dependencies are installed:

- Python 3.x
- Jupyter Notebook
- Apache Spark
- Kafka
- MariaDB
- SparkML
- Scikit-learn
- NumPy
- Pandas

Please install the necessary dependencies before running the notebook.

## Usage

Follow the steps below to run the Jupyter Notebook and perform tweet segmentation for fake news detection:

1. Clone this repository to your local machine or download the files manually.
2. Install the required dependencies as mentioned above.
3. Ensure that Apache Spark and Kafka are properly set up and configured.
4. Open the `fake_news_detection.ipynb` notebook using Jupyter Notebook.
5. Run each cell in the notebook sequentially to replicate the model training and tweet segmentation process.
6. Explore the code, comments, and visualizations to understand the underlying methodology.
7. Customize and modify the notebook as desired, experimenting with different algorithms, parameters, or feature engineering techniques.

Basic knowledge of machine learning concepts, Apache Spark, and Python programming is recommended to effectively utilize this notebook.

## Conclusion

This project showcases the application of machine learning and big data technologies to segment tweets and identify fake news. By leveraging the Random Forest Classifier, SparkML, Kafka, and MariaDB, we can efficiently process and classify incoming tweets. Feature engineering techniques like tokenization, CountVectorizer, and TF-IDF are employed to transform the raw tweet data into meaningful features.

With further improvements, such as incorporating Word2Vec and exploring other machine learning algorithms, the model's accuracy and performance can be enhanced. Additionally, connecting a BI platform to the MariaDB database can provide comprehensive insights into tweet data and support decision-making processes.

Feel free to explore the notebook, experiment with the code, and adapt it to your specific needs. For any questions or suggestions, please don't hesitate to reach out.

Happy tweet segmentation and fake news detection!
