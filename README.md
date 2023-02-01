# SMS-SPAM-FILTER

For my senior project at the Lebanese International University, I am developing a Spam Filter API to be integrated into any SMS App, platform, or firewall. The main goal of this project is to learn and have fun while doing so.

# Abstract

The project aims to train a machine learning model in Python that can classify short message service (SMS) messages as spam or not spam (ham). The model will be trained on a pre-labeled dataset of SMS messages, with the goal of creating an artificial intelligence (AI) system that can accurately identify and filter out spam messages.

The project will involve preprocessing the dataset using libraries such as pandas, NLTK, and Sklearn. The appropriate features and algorithms will be selected, using the Count Vectorization algorithm, and then training and evaluating the model using the Naive Bayes algorithm. The model will then be fine-tuned for optimal performance.

After vectorizing and training the model, it will be pickled and integrated into a flask server, allowing for the use of the POST method to post data and retrieve the model's predictions.

The ultimate goal is to make this technology usable in various systems, from mobile phones to SMS operators, firewalls, and SMS aggregation platforms.

# Acknowledgment

I would like to express my appreciation and gratitude to Dr. Fadi Yamout for his invaluable guidance and support throughout the duration of this project. I would also like to express my thanks to my friends [@princyam](https://github.com/princyam) and [@f-atwi](https://github.com/f-atwi) for their invaluable contributions in terms of new ideas, technologies and support. Their encouragement, suggestions and feedback have been instrumental in helping me achieve the best possible outcome for this project.

# Introduction

Spam messages, also known as unsolicited or unwanted messages, are a common problem in the field of electronic communication. They can take the form of email, text messages, and even phone calls. Spam messages are typically sent in bulk and often contain advertising or phishing attempts. The problem with spam is that it can be disruptive, annoying, and even harmful if users respond to them or click on links they contain.

Traditionally, spam filtering has been done using rule-based systems where messages are flagged as spam if they contain certain keywords or come from certain sources. However, as spammers become more sophisticated, rule-based systems are becoming less effective. Machine learning, on the other hand, has the ability to learn from data and adapt to changing patterns in spam messages. This makes it a more robust and effective solution for tackling the problem of spam.

Additionally, machine learning algorithms can be trained to detect spam messages with high accuracy and generalize well to new messages that have not been seen before. This is especially important as the amount of spam messages are increasing day by day and it's hard for human to go through all of them. Machine learning can also be used to identify new types of spam, which can be difficult for rule-based systems to detect.

In summary, the problem of spam messages is a significant and growing concern, and machine learning is a powerful tool that can be used to address this problem. Its ability to learn from data, adapt to changing patterns, and detect new types of spam make it an effective solution for filtering out unwanted messages.


# Training the Model

For this model we will be using the supervised learning approach. Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately. As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross-validation process. Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.

The model will be trained using the Naïve Bayes classification algorithm. Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems. It is mainly used in text classification that includes a high-dimensional training dataset. Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions. It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.

## Getting the Dataset

Getting the data and labeling them will take a long time and will require multiple people to finish it in a short period of time. So, for this Model we will be using a prelabeled dataset to train the model acquired from Kaggle.

## Importing the Necessary Libraries:

To reach the best results in a short period of time, we will be using Google Collab Platform, as Google Collab can provide us with the computational power to acquire results as fast as possible will providing the simplest and most smooth experience to work on the code.

So, we can start by importing the necessary libraries for Python to preprocess the data and train the model.

`from operator import index`

To specify if we are working with an index or not. Will be used for disregarding the index from the dataset.

`import pandas as pd`

The pandas library in python, this library us used for data manipulation. The is crucially important for this project as we are dealing with a large dataset and we will need to manipulate the data to meet our needs.

`import langdetect as detect`

We will be using only English words to train the model. So, the langdetect library can allow us to check if an SMS text is English or not. This library is a direct port of Google's language-detection library from Java to Python.

`import numpy as np`

NumPy is a Python library used for working with arrays. We will need it to convert lists and datasets to Numpy arrays to simplify the process of training the model.

`import nltk`

The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language.

  `from nltk import word\_tokenize`

Tokenizers divide strings into lists of substrings

  `from nltk.stem import WordNetLemmatizer`

Wordnet is an large, freely and publicly available lexical database for the English language aiming to establish structured semantic relationships between words. It offers lemmatization capabilities as well and is one of the earliest and most commonly used lemmatizers.

  `nltk.download('stopwords')`

A stop word is a commonly used word (such as "the", "a", "an", "in") that the model won't use and would cause extra computation.

`import sklearn`

scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support-vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. Scikit-learn is a NumFOCUS fiscally sponsored project.

  `from sklearn.feature\_extraction.text import CountVectorizer`

Convert a collection of text documents to a matrix of token counts. This implementation produces a sparse representation of the counts using scipy.sparse.csr\_matrix.

  `from sklearn.model\_selection import train\_test\_split`

The train\_test\_split() method is used to split our data into train and test sets.

  `from sklearn.naive\_bayes import MultinomialNB`

The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts.

  `from sklearn.metrics import accuracy\_score`

To check the accuracy of the model.

`import pickle`

Allows us to save the trained model into a pickle file that can be used anywhere with python.

The rest of the imports are for the sake of importing/extracting the necessary files from Google Collab or for dependencies for the libraries used.

## Preprocessing

This chapter describes the process to preprocess the data before training it so that the model can understand it.

1. Read in spam data from CSV file. We use pandas library to read the CSV file, specifiying ',' as a separator, no header, skip bad rows, and specify the encoding of the imported text to ensure a successful import.

1. Drop any "Unnamed" columns. Using pandas, we will drop any column that is unnamed since we won't be needing those columns in our model and they will cause issues. The df.drop function will drop the columns and the inplace=True parameter will ensure that the drop is directly implemented on the dataframe.

1. Define regular expression variables to remove from SMS messages. Using regular expressions or regex, we can specify the set of characters that can cause issues while training the model . So , using df.replace will passing the necessary parameters such as ' ' and regex=True, then dropping duplicates and any Nan words, this will ensure that the data is ready for the next step.

1. Remove rows with non-ASCII characters from the dataframe. To ensure that the words are readable by the model.

1. Drop non-English rows from the dataframe. Dropping rows with none English words since we are training the model only for the English language.

1. Convert all the text data into lowercase.

1. Tokenize the SMS messages in the dataframe and create a column containing the Tokenized words without the stopwords.

1. Instantiate and then assign a variable to the WordNetLemmatizer class, after that create a new column which contains the lemmatized words.

1. To ensure best results, clean the dataframe again. (This step is optional)

## Training The Model

In this part, we will be preparing then training the model.

1. We will need to initialize the count vectorizer, and then fit/transform the data.

1. We will replace the labels "spam" and "ham" (no spam) with binary 1 for "spam" and 0 for "ham".

1. Then we will split the data intro a training set and a testing set.

1. After that we will initialize the Naïve Bayas class "MultinomialNB()".

1. Then we train the model using the fit method, and check for its accuracy score using the predict method.

1. We save the model and the fitted/transformed data into a pickle file to be used for later in different python scripts.

## Preparing the API

There are several methods for making a model accessible through an API. While popular options such as Linode, AWS and other cloud platforms provide fast servers, they require financial investment. However, a cost-effective solution was discovered in the form of a free hosting service, PythonAnywhere.

PythonAnywhere website offers a platform for hosting and developing Python applications on a cloud-based server, with a user-friendly interface for managing and interacting with the server. Through the utilization of PythonAnywhere, I will be able to host the model and interact with it via the Flask web framework, making it easily accessible to users.

By creating a new web app on PythonAnywhere, we will be editing the file flask\_app.py that will contain the trained model and the flask server configuration to receive the API POST calls from user. We will be leaving the WSGI configuration as is, as we will be not needing any extra configurations than the default.

1. First upload the .pkl files to the same directory as the flask\_app.py.
2. Remove all configuration from the flask\_app.py and add the necessary import including the same imports as the trained model:


1. Create the flask app, add API keys for security, and load the pickle files:


1. Create a preprocess function to be used during the POST call:


1. Create the predict method of the API which will receive the POST API call and return the results of the model:


1. The API POST call can be done using the below curl command on Unix based systems:

`curl -X POST https://hassanshamseddine.pythonanywhere.com/predict -H 'API-Key: any available key' -H 'Content-Type: application/json' -d '{"text": "Input text here"}'`

# Use Case

The use case of the SMS spam filter API using machine learning is to integrate the trained model, which is a proof of concept, into various systems such as mobile SMS apps, SMPP aggregation systems, and firewalls. The API technology allows for the implementation of the model into these systems to filter out spam messages.

## Mobile SMS APPs

The SMS spam filter API can be integrated into mobile SMS apps to automatically filter out unwanted spam messages before they reach the user's inbox. This helps to improve the user experience by reducing the number of unwanted messages and potential scams.

## Firewall Integration

Firewalls are primarily used by network operators to secure their networks and protect against unauthorized access. By integrating the SMS spam filter API into firewalls, network operators can automatically identify and block unwanted messages before they reach the network. This can help to improve the security of the network and reduce the risk of spam messages being used to launch attacks or spread malware. Additionally, it can also help to improve the user experience by reducing the number of unwanted messages and potential scams reaching the users of the network.

## SMPP Aggregation Systems

SMPP (Short Message Peer-to-Peer) aggregation systems are used to handle high volumes of SMS traffic. By integrating the SMS spam filter API into these systems, unwanted messages can be automatically identified and blocked before they are sent to the intended recipients. This can help to improve the efficiency of the system and reduce the risk of spam messages reaching users.

## Usage Example

When using the curl command, the output should look as below:


# APIs

This appendix shows how to check for new SMS messages if they are spam or not:

`curl -X POST https://hassanshamseddine.pythonanywhere.com/predict -H 'API-Key: any available key' -H 'Content-Type: application/json' -d '{"text": "Input text here"}'`

You can add the text your wish to test in the "Input text here" parameter.

The results should look like the following:

`{"prediction":"Spam"}`

# Conclusions

The SMS spam filter project demonstrates the potential of machine learning in solving real-world problems, and the ease of implementing such solutions using the Python programming language. The use of various libraries dedicated to machine learning and the strong support from the open-source community and industry further simplifies the development process.

The SMS spam filter API, which was developed as a proof of concept, showed that AI can provide effective solutions to spam filtering. Its ability to be integrated into various modern-day systems without any issues further highlights the potential of this technology.

However, the system's performance depends on the quality and quantity of the labeled data it is trained on. With the right data, the SMS spam filter can achieve accuracy levels comparable to that of a human.

In conclusion, the SMS spam filter project has shown that AI and machine learning can effectively solve the problem of spam filtering, and the implementation of such solutions can be made simple with the use of Python and its libraries. Recommendations for future work would be to further improve the accuracy of the system and to explore additional use cases for the technology.

# References

[Pandas Documentation Link](https://pandas.pydata.org/docs/)

[Flask Documentation Link 1](https://flask.palletsprojects.com/en/2.2.x/#user-s-guide)
[Flask Documentation Link 2](https://flask.palletsprojects.com/en/2.2.x/api/)

[Sklearn Documentation Link](https://scikit-learn.org/stable/user_guide.html)

[NumPy Documentation Link](https://numpy.org/doc/)

[NLTK Documentation Link](https://www.nltk.org/)

[langdetect Documentation Link](https://pypi.org/project/langdetect/)

[pythonanywhere Documentation Link](https://help.pythonanywhere.com/pages/)

[Naïve Bayes Articles Link 1](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
[Naïve Bayes Articles Link 2](https://www.javatpoint.com/machine-learning-naive-bayes-classifier)

[Regex Documentation Link](https://docs.python.org/3/library/re.html)

Code is available on Keggle too: [Keggle Link](https://www.kaggle.com/code/hassanshamseddine/sms-spam-filter)
