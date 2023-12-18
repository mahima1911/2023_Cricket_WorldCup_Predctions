#Importing libraries

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.simplefilter("ignore")

# Importing pyspark libraries
from pyspark.sql.functions import col, regexp_replace, when
import pyspark.pandas as ps
import pandas as pd
from pyspark.sql.functions import col, lit, when, year, to_date, col, avg

from pyspark.sql.functions import col, when, col, avg
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import Row

# Initializing spark session
spark = SparkSession.builder.appName("PlayerPredict").getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
spark.sparkContext.setLogLevel("ERROR")
sqlContext = SQLContext(spark)

# Folder paths
batting_folder_path = "Full_batting_data"
bowling_folder_path = "Full_bowling_data"

# Initialize empty dictionaries to store DataFrames
batting_df = {}
bowling_df = {}

# Batting data
# Traverse through files in the batting folder
for filename in os.listdir(batting_folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(batting_folder_path, filename)
        
        # Read CSV into a DataFrame
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        
        # Store the DataFrame in the dictionary with the filename as the key
        batting_df[filename] = df


# Schema for comparison of batting
schema = ["Name", "Actual Strike Rate", "Predicted Strike Rat"]
compare_batting = spark.createDataFrame([Row("Dhoni", "4.0", "4.2")], schema=schema)

# Process batting DataFrames
for filename,df_batting_full in batting_df.items():
    df_batting_full = df_batting_full.select("Runs","Mins","BF","4s","6s","SR","Start Date")

    # Replace '-' with 0 in numeric columns
    for column_name in df_batting_full.columns:
        df_batting_full = df_batting_full.withColumn(column_name, when(col(column_name) == '-', lit(0)).otherwise(col(column_name)))
    df_batting_year = df_batting_full.withColumn("Year", col("Start Date").substr(-4, 4))

    # Clean the "Runs" column
    df_batting_year = df_batting_year.withColumn("Runs", when(col("Runs").like("%*"), regexp_replace(col("Runs"), "\\*", "").cast("int")).otherwise(col("Runs")))
    df_batting_year = df_batting_year.withColumn("Runs", when(col("Runs").isin("DNB", "TDNB"), 0).otherwise(col("Runs").cast("int")))
    numeric_columns = ["Runs", "Mins", "BF", "4s", "6s", "SR"]
    for column in numeric_columns:
        df_batting_year = df_batting_year.withColumn(column, df_batting_year[column].cast("double"))

    # Calculate average values by grouping by the "Year" column
    avg_batting_df = df_batting_year.groupBy("Year").agg(
        avg("Runs").alias("avg_runs"),
        avg("Mins").alias("avg_mins"),
        avg("BF").alias("avg_BF"),
        avg("4s").alias("avg_4s"),
        avg("6s").alias("avg_6s"),
        avg("SR").alias("avg_SR")
    )

    # Select the relevant columns as features
    features_columns = ["avg_runs", "avg_mins", "avg_BF", "avg_4s", "avg_6s", "avg_SR"]

    # Filter only the rows where avg_SR is not null (exclude the row for 2023)
    train_data = avg_batting_df.filter(col("Year") != 2023).select(*features_columns)

    avg_batting_df_year = avg_batting_df.filter(col("Year") == 2023)

    actual_sr = avg_batting_df_year.first()["avg_SR"]
    actual_sr = round(abs(actual_sr),2)

    # Create a vector assembler to assemble features into a vector
    assembler = VectorAssembler(inputCols=features_columns[:-1], outputCol="features")

    # Transform the data using the vector assembler
    train_data = assembler.transform(train_data)

    # Create a Linear Regression model
    lr = LinearRegression(featuresCol="features", labelCol="avg_SR",regParam=0.1)

    # Fit the model to the training data
    model = lr.fit(train_data)

    # Prepare the features for the prediction (for the year 2023)
    prediction_features = assembler.transform(avg_batting_df.filter(col("Year") == 2023).select(*features_columns[:-1]))

    # Make predictions for the year 2023
    predictions_batting = model.transform(prediction_features)
    pred_sr = predictions_batting.first()["prediction"]
    pred_sr = round(abs(pred_sr),2)

    batsman_name = filename.split("_")[-2]

    print("Linear Regression model predicted ",pred_sr," Strike rate for ",batsman_name)

    compare_batting = compare_batting.union(spark.createDataFrame([(batsman_name, actual_sr, pred_sr)], schema=schema))
    # Show the predictions

compare_batting = compare_batting.filter(col("Name") != "Dhoni")
print("Comparison of Actual Batting Strike rate vs. Predicted Strike rate:")
compare_batting.show()

####################################################

# Bowling data
# Traverse through files in the batting folder
for filename in os.listdir(bowling_folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(bowling_folder_path, filename)
        
        # Read CSV into a DataFrame
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        
        # Store the DataFrame in the dictionary with the filename as the key
        bowling_df[filename] = df

# Schema for comparison of bowling
schema = ["Name", "Actual Economy", "Predicted Economy"]
compare_bowling = spark.createDataFrame([Row("Dhoni", "4.0", "4.2")], schema=schema)

# Process bowling DataFrames
for filename,df_bowling_full in bowling_df.items():
    df_bowling_full = df_bowling_full.select("Overs","Mdns","Runs","Wkts","Econ","Start Date")

    # Replace '-' with 0 in numeric columns
    for column_name in df_bowling_full.columns:
        df_bowling_full = df_bowling_full.withColumn(column_name, when(col(column_name) == '-', lit(0)).otherwise(col(column_name)))

    df_bowling_year = df_bowling_full.withColumn("Year", col("Start Date").substr(-4, 4))

    # Clean the "Runs" column
    df_bowling_year = df_bowling_year.withColumn("Overs", when(col("Overs").like("%*"), regexp_replace(col("Overs"), "\\*", "").cast("int")).otherwise(col("Overs")))
    df_bowling_year = df_bowling_year.withColumn("Overs", when(col("Overs").isin("DNB", "TDNB"), 0).otherwise(col("Overs").cast("int")))
    numeric_columns_bowling = ["Overs", "Mdns", "Runs", "Wkts", "Econ"]
    for column in numeric_columns_bowling:
        df_bowling_year = df_bowling_year.withColumn(column, df_bowling_year[column].cast("double"))

    # Calculate average values by grouping by the "Year" column
    avg_bowling_df = df_bowling_year.groupBy("Year").agg(
        avg("Overs").alias("avg_Overs"),
        avg("Mdns").alias("avg_Mdns"),
        avg("Runs").alias("avg_Runs"),
        avg("Wkts").alias("avg_Wkts"),
        avg("Econ").alias("avg_Econ")
    )
    features_columns = ["avg_Overs" ,"avg_Mdns","avg_Runs","avg_Wkts","avg_Econ"]

    # Filter only the rows where avg_SR is not null (exclude the row for 2023)
    train_data = avg_bowling_df.filter(col("Year") != 2023).select(*features_columns)
    avg_bowling_df_year = avg_bowling_df.filter(col("Year") == 2023)

    actual_econ = avg_bowling_df_year.first()["avg_Econ"]
    actual_econ = round(abs(actual_econ),2)

    # Create a vector assembler to assemble features into a vector
    assembler = VectorAssembler(inputCols=features_columns[:-1], outputCol="features")

    # Transform the data using the vector assembler
    train_data = assembler.transform(train_data)

    # Create a Linear Regression model
    lr = LinearRegression(featuresCol="features", labelCol="avg_Econ",regParam=0.1)

    # Fit the model to the training data
    model = lr.fit(train_data)

    # Prepare the features for the prediction (for the year 2023)
    prediction_features = assembler.transform(avg_bowling_df.filter(col("Year") == 2023).select(*features_columns[:-1]))

    # Make predictions for the year 2023
    predictions_bowling = model.transform(prediction_features)
    pred_econ = predictions_bowling.first()["prediction"]
    pred_econ = round(abs(pred_econ),2)


    bowler_name = filename.split("_")[-2]

    print("Linear Regression model predicted ",pred_econ,"Economy for ",bowler_name)

    compare_bowling = compare_bowling.union(spark.createDataFrame([(bowler_name, actual_econ, pred_econ)], schema=schema))
    # Show the predictions

compare_bowling = compare_bowling.filter(col("Name") != "Dhoni")
print("Comparison of Actual Bowling Economy vs. Predicted Economy:")
compare_bowling.show()


