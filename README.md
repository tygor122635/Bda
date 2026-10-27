BTML4501 — Big Data Analytics (BDA) Lab
Selected Experiments: 4, 5, 6, 7, 9, and 11

This repository contains Python demonstrations for selected Big Data Analytics laboratory experiments.

Table of Contents
Experiment 4 — MapReduce Word Count
Experiment 5 — Mining Healthcare Data
Experiment 6 — HBase Table Management
Experiment 7 — HBase Data Operations
Experiment 9 — SparkConf Attributes
Experiment 11 — Spark MLlib Logistic Regression
Requirements
Python 3
pandas for Experiment 5
pyspark for Experiments 9 and 11
Install the Python dependencies:

pip install pandas pyspark
Experiments 4, 6, and 7 use standard Python libraries and do not require additional packages.

Experiment 4 — MapReduce Word Count
Aim: Count each word using the MapReduce stages: Map → Shuffle/Sort → Reduce.

How it works
Map: Convert each word into a (word, 1) pair.
Shuffle/Sort: Group values belonging to the same word.
Reduce: Add the values to calculate each word's total count.
Code
from collections import defaultdict
import re

text_data = """Hadoop is an open source framework
Hadoop stores data in HDFS
MapReduce processes data in Hadoop
Big data needs Hadoop and Spark"""

print("INPUT:\n", text_data)

# MAP: output (word, 1)
mapped = []
for line in text_data.lower().splitlines():
    for word in re.findall(r"[a-z0-9]+", line):
        mapped.append((word, 1))
print("\nMAP output (first 15):", mapped[:15])

# SHUFFLE/SORT: group values by word
groups = defaultdict(list)
for word, value in mapped:
    groups[word].append(value)

print("\nSHUFFLE/SORT:")
for word in sorted(groups):
    print(word, groups[word])

# REDUCE: add values
counts = {word: sum(values) for word, values in groups.items()}
print("\nFINAL COUNTS:")
for word in sorted(counts):
    print(f"{word:<15} {counts[word]}")
Experiment 5 — Mining Healthcare Data
Aim: Calculate patient counts by disease and gender, average bill per disease, and average stay per department.

The demonstration creates a small sample healthcare dataset and loads it into a pandas DataFrame.

Code
import pandas as pd
from io import StringIO

csv_data = """patient_id,age,gender,disease,department,stay_days,bill_amount
P001,45,M,Diabetes,Endocrinology,4,18000
P002,62,F,Hypertension,Cardiology,3,15000
P003,29,F,Asthma,Pulmonology,2,9000
P004,55,M,Heart Disease,Cardiology,7,85000
P005,38,F,Diabetes,Endocrinology,5,21000
P006,70,M,Hypertension,Cardiology,4,17000
P007,33,M,Asthma,Pulmonology,3,11000
P008,48,F,Heart Disease,Cardiology,8,92000
P009,51,M,Diabetes,Endocrinology,6,24000
P010,26,F,Fracture,Orthopedics,5,40000
P011,64,M,Heart Disease,Cardiology,9,98000
P012,41,F,Asthma,Pulmonology,2,8000
P013,58,F,Hypertension,Cardiology,3,14000
P014,35,M,Fracture,Orthopedics,6,46000
P015,67,F,Diabetes,Endocrinology,5,22000
P016,31,M,Fracture,Orthopedics,4,38000
P017,59,M,Hypertension,Cardiology,5,19000
P018,44,F,Asthma,Pulmonology,3,10000
P019,72,F,Heart Disease,Cardiology,10,105000
P020,36,M,Diabetes,Endocrinology,4,17000
"""

patients = pd.read_csv(StringIO(csv_data))
print(patients.head())
print("Total records:", len(patients))
Note: The code above loads and displays the sample records. The stated analysis goals—counts by disease and gender, average bill by disease, and average stay by department—can be calculated with pandas grouping and aggregation.

Experiment 6 — HBase Table Management (Simulation)
Aim: Simulate the HBase operations create, list, exists, describe, disable, enable, alter, and drop.

This is a Python simulation using a dictionary; it does not connect to a real HBase cluster.

Code
tables = {}

def create_table(name, families):
    tables[name] = {"families": list(families), "enabled": True, "rows": {}}
    print("Created table:", name)

def list_tables():
    print("Tables:", list(tables))

def exists(name):
    print("Exists:", name in tables)

def describe(name):
    print(tables.get(name, "Table does not exist"))

def disable(name):
    tables[name]["enabled"] = False
    print(name, "disabled")

def enable(name):
    tables[name]["enabled"] = True
    print(name, "enabled")

def add_family(name, family):
    if tables[name]["enabled"]:
        print("Disable the table before altering it.")
    else:
        tables[name]["families"].append(family)
        print("Added family:", family)

def drop(name):
    if tables[name]["enabled"]:
        print("Disable the table before dropping it.")
    else:
        del tables[name]
        print("Dropped table:", name)

create_table("patient", ["info", "visit"])
list_tables()
exists("patient")
describe("patient")
disable("patient")
add_family("patient", "billing")
describe("patient")
enable("patient")
disable("patient")
drop("patient")
exists("patient")
Experiment 7 — HBase Data Operations (Simulation)
Aim: Practise put, get, scan, count, update, delete a cell, delete a row, and truncate.

This experiment represents HBase-style rows and columns with nested Python dictionaries. It is a simulation, not a connection to HBase.

Code
student = {}

def put(row, column, value):
    student.setdefault(row, {})[column] = value
    print(f"Stored {row} / {column} = {value}")

def get(row):
    print(row, student.get(row, "ROW NOT FOUND"))

def scan():
    for row in sorted(student):
        print(row, student[row])

def delete_cell(row, column):
    if row in student:
        student[row].pop(column, None)
        if not student[row]:
            del student[row]
    print("Deleted cell:", row, column)

def delete_row(row):
    student.pop(row, None)
    print("Deleted row:", row)

put("101", "personal:name", "Ravi")
put("101", "personal:city", "Hyderabad")
put("101", "marks:bda", "88")
put("102", "personal:name", "Sita")
put("102", "personal:city", "Chennai")
put("102", "marks:bda", "92")
put("103", "personal:name", "Anil")
put("103", "marks:bda", "75")

print("\nGET row 101")
get("101")

print("\nSCAN")
scan()
print("COUNT:", len(student))

print("\nUPDATE row 103")
put("103", "marks:bda", "80")
get("103")

print("\nDELETE one cell")
delete_cell("102", "personal:city")
get("102")

print("\nDELETE one row")
delete_row("103")
print("COUNT:", len(student))

print("\nTRUNCATE all rows")
student.clear()
scan()
print("COUNT after truncate:", len(student))
Experiment 9 — SparkConf Attributes
Aim: Set and read Spark application settings and create a SparkContext.

Code
from pyspark import SparkConf, SparkContext

conf = (
    SparkConf()
    .setAppName("SparkConfDemo")
    .setMaster("local[2]")
    .set("spark.executor.memory", "1g")
    .set("spark.driver.memory", "1g")
    .set("spark.sql.shuffle.partitions", "4")
)

conf.setIfMissing("spark.ui.enabled", "false")
conf.setExecutorEnv("LAB_NAME", "BDA")

print("App name:", conf.get("spark.app.name"))
print("Master:", conf.get("spark.master"))
print("Executor memory:", conf.get("spark.executor.memory"))
print("Has driver memory:", conf.contains("spark.driver.memory"))
print("Missing key default:", conf.get("spark.not.set", "default-value"))

print("\nAll settings:")
for key, value in sorted(conf.getAll()):
    print(key, "=", value)

sc = SparkContext.getOrCreate(conf=conf)
sc.setLogLevel("ERROR")
print("Context app name:", sc.appName)
print("Context master:", sc.master)
print("Default parallelism:", sc.defaultParallelism)
print("Sum 1..100:", sc.parallelize(range(1, 101)).sum())
sc.stop()
Key concepts
SparkConf stores configuration settings for a Spark application.
setAppName() sets the application name.
setMaster("local[2]") runs Spark locally using two worker threads.
SparkContext is the entry point for Spark's low-level functionality.
Experiment 11 — Spark MLlib: Logistic Regression
Aim: Build a machine-learning pipeline using VectorAssembler, StandardScaler, and LogisticRegression, then evaluate AUC and accuracy.

Code
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("BDA_MLlib_LogisticRegression")
    .master("local[*]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

rows = [
    (25, 22.0, 85, 0), (30, 24.5, 90, 0),
    (35, 27.0, 100, 0), (28, 23.0, 88, 0),
    (40, 26.5, 95, 0), (32, 25.0, 92, 0),
    (45, 28.0, 105, 0), (38, 24.0, 98, 0),
    (50, 31.0, 150, 1), (55, 33.5, 165, 1),
    (60, 30.0, 140, 1), (48, 34.0, 155, 1),
    (52, 29.5, 148, 1), (58, 35.0, 170, 1),
    (62, 32.0, 160, 1), (47, 30.5, 145, 1),
    (42, 27.5, 110, 0), (36, 31.5, 135, 1),
    (44, 29.0, 125, 1), (29, 23.5, 87, 0),
]

df_ml = spark.createDataFrame(rows, ["age", "bmi", "glucose", "label"])
train, test = df_ml.randomSplit([0.75, 0.25], seed=42)
print("Training rows:", train.count(), "| Test rows:", test.count())

assembler = VectorAssembler(
    inputCols=["age", "bmi", "glucose"],
    outputCol="raw_features",
)
scaler = StandardScaler(inputCol="raw_features", outputCol="features")
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    regParam=0.01,
)

pipeline = Pipeline(stages=[assembler, scaler, lr])
model = pipeline.fit(train)

pred = model.transform(test)
pred.select("age", "bmi", "glucose", "label", "prediction").show()

auc = BinaryClassificationEvaluator(
    labelCol="label", metricName="areaUnderROC"
).evaluate(pred)
acc = MulticlassClassificationEvaluator(
    labelCol="label", metricName="accuracy"
).evaluate(pred)

print(f"Area under ROC: {auc:.3f}")
print(f"Accuracy: {acc:.3f}")

lr_model = model.stages[-1]
print("Coefficients:", [round(float(c), 3) for c in lr_model.coefficients])
print("Intercept:", round(float(lr_model.intercept), 3))

spark.stop()
Pipeline overview
VectorAssembler combines the input columns into a feature vector.
StandardScaler scales the feature vector.
LogisticRegression trains a binary classification model.
The model predicts labels for the test dataset.
AUC and accuracy are calculated to evaluate performance.
Important: The healthcare and ML records in these examples are small, illustrative datasets for lab practice. They are not intended for real clinical decision-making.

Running the Experiments
Save each experiment's code in a separate Python file, for example:

experiment_4_mapreduce.py
experiment_5_healthcare.py
experiment_6_hbase_tables.py
experiment_7_hbase_operations.py
experiment_9_sparkconf.py
experiment_11_logistic_regression.py
Run a script with:

python experiment_4_mapreduce.py
For Spark experiments, ensure Java and PySpark are correctly installed and configured in your environment.

Learning Outcomes
After completing these experiments, you should be able to:

Explain the MapReduce word-count workflow.
Load and inspect healthcare data with pandas.
Simulate basic HBase table and row operations.
Configure a local Apache Spark application.
Build and evaluate a basic binary-classification pipeline with Spark MLlib.


