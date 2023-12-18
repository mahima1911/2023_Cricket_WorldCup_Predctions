# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, lit, when, year, to_date, col, avg
from pyspark.sql import SparkSession, SQLContext

# Initialize Spark session
spark = SparkSession.builder.appName("PlayerData").getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
sqlContext = SQLContext(spark)

# Folder path
folder_path = "Full_batting_data"

# Initialize an empty dictionary to store dataframes
dataframes = {}

# Traverse through files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # Read CSV into a DataFrame
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        
        # Store the DataFrame in the dictionary with the filename as the key
        dataframes[filename] = df


# Plotting
for filename, player_spark_df in dataframes.items():

    # Iterate through all columns and replace '-' with 0
    for column_name in player_spark_df.columns:
        player_spark_df = player_spark_df.withColumn(column_name, when(col(column_name) == '-', lit(0)).otherwise(col(column_name)))
    df_with_year = player_spark_df.withColumn("Year", year(to_date(col("Start Date"), "dd MMM yyyy")))
    result_df = df_with_year.groupBy("Year").agg(avg("SR").alias("Avg_SR"))

    # df_with_year.printSchema()
    result_df.show()

    result_df = result_df.toPandas()

    plt.figure(figsize=(10, 6))
    plt.bar(result_df['Year'], result_df['Avg_SR'])
    plt.title(filename[:-4])  # Remove ".csv" from the filename for the title
    plt.xlabel('Year')
    plt.ylabel('Avg_SR')
    plt.show()
