from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, when, month
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("HealthRiskPrediction").getOrCreate()

#I'm using the small data set because the large dataset's features did not correlate to is_anomaly. I take a single entry from the 
#large dataset, because none of the small dataset's entries are from London, and this caused an error in my Location Indexer
data_path = "../health_events_dataset.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
large_df = spark.read.csv("../1m_health_events_dataset.csv", header=True,inferSchema=True)
#ADD DIFFERENT FILTER TO CHANGE DATA-SET
large_df = large_df.filter(large_df.Location == "London").limit(1)
df = df.union(large_df)

# Find the Month, encode the eventType, Location, and Severity
df = df.withColumn("Timestamp", unix_timestamp(col("Timestamp")).cast("timestamp"))
df = df.withColumn("Month", month(col("Timestamp")))

indexers = [
    StringIndexer(inputCol="EventType", outputCol="EventTypeIndex"),
    StringIndexer(inputCol="Location", outputCol="LocationIndex"),
]

# save the indexer for use on the stream
for indexer in indexers:
    indexer_model = indexer.fit(df)
    indexer_model.save(f"{indexer.getInputCol()}_indexer")
    df = indexer_model.transform(df)

df = df.withColumn(
    "SeverityNumeric",
    when(col("Severity") == "low", 0)
    .when(col("Severity") == "medium", 1)
    .when(col("Severity") == "high", 2)
    .otherwise(None)
)

# create feature assembler
feature_cols = ["EventTypeIndex", "LocationIndex", "SeverityNumeric", "Month"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Print statements
print(f"Training set size: {train.count()}")
print(f"Test set size: {test.count()}")
print("Training set anomaly breakdown:")
train.groupBy("Is_Anomaly").count().show()
print("Test set anomaly breakdown:")
test.groupBy("Is_Anomaly").count().show()


#Train the model
rf = RandomForestClassifier(labelCol="Is_Anomaly", featuresCol="features", numTrees=100)
model = rf.fit(train)
predictions = model.transform(test)

# Evaluate model performance
evaluator = MulticlassClassificationEvaluator(
    labelCol="Is_Anomaly", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Test set accuracy: {accuracy:.2f}")

# Count correctly predicted anomalies
correct_anomalies = predictions.filter((col("Is_Anomaly") == 1) & (col("prediction") == 1)).count()
total_anomalies = predictions.filter(col("Is_Anomaly") == 1).count()
print(f"Correctly predicted anomalies: {correct_anomalies} out of {total_anomalies}")

# Count overall predictions
total_predictions = predictions.count()
print(f"Total test set size: {total_predictions}")

#Save Model
model.save("rf_model_health_risk")
