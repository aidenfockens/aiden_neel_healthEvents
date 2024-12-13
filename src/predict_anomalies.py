from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, month, when, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexerModel
import json


spark = SparkSession.builder \
    .appName("Proj2Part3") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .getOrCreate()

df = spark.read.format("kafka") \
               .option("kafka.bootstrap.servers", "3.80.138.59:9092") \
               .option("subscribe", "health_events") \
               .option("startingOffsets", "earliest") \
               .option("endingOffsets", "latest").load()

# Decode binary data to string
def decode(bs, enc="utf-8"):
    if bs is not None:
        try:
            return bytes(bs).decode(enc)
        except:
            return None

def extract_field(json_str, field_name):
    if json_str is not None:
        try:
            return json.loads(json_str).get(field_name)
        except json.JSONDecodeError:
            return None
    return None

decode_udf = udf(decode, StringType())
df = df.withColumn("value_decoded", decode_udf("value"))

# Create features
fields_to_extract = ["EventType", "Timestamp", "Location", "Severity", "Details"]
for field in fields_to_extract:
    extract_column_udf = udf(lambda json_str: extract_field(json_str, field), StringType())
    df = df.withColumn(field, extract_column_udf(col("value_decoded")))
df = df.withColumn("Timestamp", to_timestamp(col("Timestamp")))
selected_df = df.select(fields_to_extract)
sorted_df = selected_df.orderBy(col("Timestamp").asc())

# Only take 5000 data points
recent_5000_df = sorted_df.limit(5000)  
recent_5000_df = recent_5000_df.withColumn("Month", month(col("Timestamp")))

# Create features
event_type_indexer = StringIndexerModel.load("EventType_indexer")
location_indexer = StringIndexerModel.load("Location_indexer")
recent_5000_df = event_type_indexer.transform(recent_5000_df)
recent_5000_df = location_indexer.transform(recent_5000_df)
recent_5000_df = recent_5000_df.withColumn(
    "SeverityNumeric",
    when(col("Severity") == "low", 0)
    .when(col("Severity") == "medium", 1)
    .when(col("Severity") == "high", 2)
    .otherwise(None)
)

# Create Assembler
feature_cols = ["EventTypeIndex", "LocationIndex", "SeverityNumeric", "Month"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
recent_5000_df = assembler.transform(recent_5000_df)

# Load the trained Random Forest model
loaded_model = RandomForestClassificationModel.load("rf_model_health_risk")

# Make predictions
predictions = loaded_model.transform(recent_5000_df)

# Count anomalies and non-anomalies
anomalies_count = predictions.filter(col("prediction") == 1).count()
non_anomalies_count = predictions.filter(col("prediction") == 0).count()
print(f"Number of anomalies detected: {anomalies_count}")
print(f"Number of non-anomalies detected: {non_anomalies_count}")

# Show examples of anomalies
print("Examples of anomalies:")
predictions.filter(col("prediction") == 1).select(
    "EventType", "Timestamp", "Location", "Severity", "Details"
).show(5, truncate=False)

# Show examples of non-anomalies
print("Examples of non-anomalies:")
predictions.filter(col("prediction") == 0).select(
    "EventType", "Timestamp", "Location", "Severity", "Details"
).show(5, truncate=False)
