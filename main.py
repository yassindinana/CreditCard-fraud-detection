import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt

# Initialize SparkSession
spark = SparkSession.builder.appName('CreditCardFraudDetection').getOrCreate()

# Load the credit card fraud dataset
df = spark.read.csv('data.csv', header=True, inferSchema=True)

# Calculate the number of fraud and non-fraud transactions
counts = df.groupBy('Class').count().orderBy('Class').collect()
fraud_count = counts[1]['count']
non_fraud_count = counts[0]['count']

# Print the number of fraud and non-fraud transactions
print(f'Number of Fraud Transactions: {fraud_count}')
print(f'Number of Non-Fraud Transactions: {non_fraud_count}')

# Split the dataset into training and testing sets
(trainingData, testData) = df.randomSplit([0.7, 0.3], seed=1234)

# Prepare the features for machine learning
assembler = VectorAssembler(inputCols=[col for col in df.columns if col not in ['Time', 'Class']], outputCol='features')
trainingData = assembler.transform(trainingData)
testData = assembler.transform(testData)

# Train a random forest classifier
rf = RandomForestClassifier(labelCol='Class', featuresCol='features', numTrees=100)
rfModel = rf.fit(trainingData)

# Make predictions on the testing data
predictions = rfModel.transform(testData)

# Evaluate the performance of the model
evaluator = BinaryClassificationEvaluator(labelCol='Class')
auc = evaluator.evaluate(predictions)
print(f'AUC: {auc}')

# Calculate the number of true positive, false positive, true negative, and false negative predictions
true_positive = predictions.filter((predictions.Class == 1) & (predictions.prediction == 1)).count()
false_positive = predictions.filter((predictions.Class == 0) & (predictions.prediction == 1)).count()
true_negative = predictions.filter((predictions.Class == 0) & (predictions.prediction == 0)).count()
false_negative = predictions.filter((predictions.Class == 1) & (predictions.prediction == 0)).count()

# Print the confusion matrix
print(f'True Positive: {true_positive}')
print(f'False Positive: {false_positive}')
print(f'True Negative: {true_negative}')
print(f'False Negative: {false_negative}')

# Create a bar plot of transaction counts
labels = ['Fraud', 'Non-Fraud']
counts = [fraud_count, non_fraud_count]
fig, ax1 = plt.subplots()

# Plot the fraud transaction counts on a separate y-axis scale
ax2 = ax1.twinx()
ax2.set_ylim([0, fraud_count])
ax2.set_ylabel('Fraud Transactions')

# Plot the non-fraud transaction counts on the left y-axis
ax1.bar(labels, counts)
ax1.set_ylabel('Non-Fraud Transactions')
plt.title('Transaction Counts by Class')
plt.show()
