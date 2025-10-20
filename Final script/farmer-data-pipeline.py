# glue_data_pipeline_fixed.py

import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# -----------------------------
# Initialize Glue Context
# -----------------------------
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# -----------------------------
# 1. Read Raw CSV from S3
# -----------------------------
raw_df = spark.read.option("header", True).csv("s3a://farmer-raw-data/farmer_suicide_large_realistic_train.csv")

# -----------------------------
# 2. Convert numeric columns to DoubleType
# -----------------------------
for col in raw_df.columns:
    try:
        raw_df = raw_df.withColumn(col, raw_df[col].cast(DoubleType()))
    except:
        pass  # skip if cannot cast (categorical columns)

# -----------------------------
# 3. Fill missing numeric values
# -----------------------------
numeric_cols = [f.name for f in raw_df.schema.fields if isinstance(f.dataType, DoubleType)]
for col in numeric_cols:
    mean_val = raw_df.select(F.mean(F.col(col))).first()[0]
    raw_df = raw_df.withColumn(col, F.when(F.col(col).isNull(), mean_val).otherwise(F.col(col)))

# -----------------------------
# 4. Encode categorical columns (one-hot)
# -----------------------------
cat_cols = [f.name for f in raw_df.schema.fields if str(f.dataType) == "StringType"]
for col in cat_cols:
    raw_df = raw_df.fillna({col: "Unknown"})

# Use StringIndexer + OneHotEncoder
from pyspark.ml.feature import StringIndexer, OneHotEncoder

for col in cat_cols:
    indexer = StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep")
    raw_df = indexer.fit(raw_df).transform(raw_df)
    encoder = OneHotEncoder(inputCols=[col + "_index"], outputCols=[col + "_encoded"])
    raw_df = encoder.fit(raw_df).transform(raw_df)

# -----------------------------
# 5. Drop original categorical columns
# -----------------------------
raw_df = raw_df.drop(*cat_cols)
raw_df = raw_df.drop(*[col + "_index" for col in cat_cols])

# -----------------------------
# 6. Rename columns to remove spaces
# -----------------------------
raw_df = raw_df.toDF(*[c.replace(" ", "_") for c in raw_df.columns])

# -----------------------------
# 7. Save processed data to S3
# -----------------------------
raw_df.write.mode("overwrite").option("header", True)\
    .csv("s3a://farmer-processed-data/farmer_suicide_processed.csv")

print("✅ Glue ETL job completed successfully!")
