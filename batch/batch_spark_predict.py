"""
Spark-based batch prediction for breast cancer classification.

Uses PySpark for data loading and orchestration,
and scikit-learn for model inference.
"""

from pyspark.sql import SparkSession
import os
import joblib
import pandas as pd
import logging

spark = SparkSession.builder.appName("BreastCancerBatchPrediction").getOrCreate()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV_PATH = os.path.join(BASE_DIR, "batch", "sample_input.csv")

spark_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(INPUT_CSV_PATH)

spark_df.printSchema()
spark_df.show(5)

REQUIRED_COLUMNS = {"mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness", "mean_compactness",
                    "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension", "radius_error",
                    "texture_error", "perimeter_error", "area_error", "smoothness_error", "compactness_error",
                    "concavity_error", "concave_points_error", "symmetry_error", "fractal_dimension_error",
                    "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness",
                    "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry",
                    "worst_fractal_dimension"}

spark_columns = set(spark_df.columns)
missing_cols = REQUIRED_COLUMNS - spark_columns

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("✅ Spark schema validation passed")

from pyspark.sql.functions import col, sum as spark_sum

null_counts = spark_df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in spark_df.columns
])

null_counts.show()

print("✅ Spark null check completed")

pdf = spark_df.toPandas()
print("✅ Converted Spark DataFrame to Pandas")

FEATURE_NAME_MAPPING = {
    "mean_radius": "mean radius",
    "mean_texture": "mean texture",
    "mean_perimeter": "mean perimeter",
    "mean_area": "mean area",
    "mean_smoothness": "mean smoothness",
    "mean_compactness": "mean compactness",
    "mean_concavity": "mean concavity",
    "mean_concave_points": "mean concave points",
    "mean_symmetry": "mean symmetry",
    "mean_fractal_dimension": "mean fractal dimension",

    "radius_error": "radius error",
    "texture_error": "texture error",
    "perimeter_error": "perimeter error",
    "area_error": "area error",
    "smoothness_error": "smoothness error",
    "compactness_error": "compactness error",
    "concavity_error": "concavity error",
    "concave_points_error": "concave points error",
    "symmetry_error": "symmetry error",
    "fractal_dimension_error": "fractal dimension error",

    "worst_radius": "worst radius",
    "worst_texture": "worst texture",
    "worst_perimeter": "worst perimeter",
    "worst_area": "worst area",
    "worst_smoothness": "worst smoothness",
    "worst_compactness": "worst compactness",
    "worst_concavity": "worst concavity",
    "worst_concave_points": "worst concave points",
    "worst_symmetry": "worst symmetry",
    "worst_fractal_dimension": "worst fractal dimension"
}


pdf.rename(columns=FEATURE_NAME_MAPPING, inplace=True)
print("✅ Feature names mapped for ML model")

MODEL_PATH = os.path.join(BASE_DIR, "models", "cancer_pipeline.joblib")

model = joblib.load(MODEL_PATH)
print("✅ ML model loaded")

predictions = model.predict(pdf)
probabilities = model.predict_proba(pdf)

label_map = {0: "malignant", 1: "benign"}

pdf["prediction_code"] = predictions
pdf["prediction"] = [label_map[p] for p in predictions]
pdf["probability"] = probabilities.max(axis=1)

output_path = os.path.join(BASE_DIR, "batch", "spark_predictions_output.csv")
pdf.to_csv(output_path, index=False)

print(f"✅ Spark batch predictions saved to {output_path}")
