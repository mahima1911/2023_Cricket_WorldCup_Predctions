import pyspark.pandas as ps
import pandas as pd
from pyspark.sql.functions import col, lit, when, year, to_date, col, avg
from pyspark.sql import SparkSession, SQLContext

spark = SparkSession.builder.appName("PlayerData").getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
def getBowlingData(player_id):


  sqlContext = SQLContext(spark)
  player_df = pd.read_html('https://stats.espncricinfo.com/ci/engine/player/'+player_id+'.html?class=2;template=results;type=bowling;view=innings')[3]
  # print(player_df)
  player_spark_df = sqlContext.createDataFrame(player_df)
  # Iterate through all columns and replace '-' with 0
  for column_name in player_spark_df.columns:
      player_spark_df = player_spark_df.withColumn(column_name, when(col(column_name) == '-', lit(0)).otherwise(col(column_name)))
  # player_spark_df.show()
  df_with_year = player_spark_df.withColumn("Year", year(to_date(col("Start Date"), "dd MMM yyyy")))
  result_df = df_with_year.groupBy("Year").agg(avg("Econ").alias("Avg_Econ"))

  # df_with_year.printSchema()
  result_df.show()
  return result_df