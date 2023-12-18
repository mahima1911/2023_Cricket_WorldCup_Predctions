import pandas as pd
from pyspark.sql.functions import col, lit, when, year, to_date, col, avg
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#start a new spark session
spark = SparkSession.builder.appName("project").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
spark.sparkContext.setLogLevel("ERROR")
sqlContext = SQLContext(spark)

# Read the data from "World_cup_2023.csv" into the 'World_cup' DataFrame
WCdata = pd.read_csv("World_cup_2023.csv")

# Read the data from "results.csv" into the 'results' DataFrame
results = pd.read_csv("results.csv")

# Read the data from "icc_ranking.csv" into the 'results' DataFrame
ranking = pd.read_csv("Icc_ranking.csv")


WCdata_spark_df = sqlContext.createDataFrame(WCdata)
print("Displaying World Cup 2023 data")
WCdata_spark_df.show()

ranking_spark_df = sqlContext.createDataFrame(ranking)
print("Displaying Ranking of teams in World Cup ")
ranking_spark_df.show()

results_spark_df = sqlContext.createDataFrame(results)
print("Dataset")
results_spark_df.show()

#  Drop rows where 'Winner' is 'Match abandoned' or 'No result'
results_spark_df = results_spark_df.filter((col('Winner') != 'Match abandoned') & (col('Winner') != 'No result'))

# Show the updated DataFrame
print("Dataset after dropping rows where 'Winner' is 'Match abandoned' or 'No result")
results_spark_df.show()

# Filter the DataFrame to include rows where India played as either Team_1 or Team_2
india_spark_df = results_spark_df.filter((col('Team_1') == 'India') | (col('Team_2') == 'India'))

print("Matches Team India played")
india_spark_df.show()

# Filtering the 'India' dataframe to create a new dataframe 'India_win' containing rows where the 'Winner' column is 'India'.
India_spark_win = india_spark_df.filter((col('Winner') == 'India'))

# Display the first few rows of the 'India' DataFrame
print("Matches Team India won")
India_spark_win.show()

excluded_value = 'India'
filtered_spark_df = India_spark_win.filter((col('Team_2') != excluded_value))

# Counting the occurrences of each value in the filtered DataFrame's 'Team_2' column
value_counts = filtered_spark_df.groupBy('Team_2').count()

# Exclude Team India's name
excluded_value = 'India'

# Filtering out rows with the excluded value
filtered_spark_df = India_spark_win.filter((col('Team_1') != excluded_value))

# Counting the occurrences of each value in the filtered DataFrame's 'Team_2' column.
value_counts = filtered_spark_df.groupBy('Team_2').count()

# results_spark_df.show()

worldcup_teams = ['England','Bangladesh','South Africa','India','Pakistan','Australia','New Zealand','Afghanistan']

# Filtering matches involving only teams in the 'worldcup_teams' list
df_teams_1 = results_spark_df.filter(col('Team_1').isin(worldcup_teams))
df_teams_2 = results_spark_df.filter(col('Team_2').isin(worldcup_teams))

# Combining the filtered DataFrames
df_teams = df_teams_1.union(df_teams_2)

# Removing duplicate rows
df_teams = df_teams.dropDuplicates()

# Counting the number of rows in the resulting DataFrame
count = df_teams.count()

# Displaying the first few rows of the "df_teams" DataFrame.
# df_teams.show()

columns_to_drop = ['Date', 'Margin', 'Ground']
df_teams_all = df_teams.drop(*columns_to_drop)

#  Displaying the first few rows of the DataFrame
print("Dataset after dropping uninformative columns")
df_teams_all.show()

indexer_team_1 = StringIndexer(inputCol="Team_1", outputCol="Team_1_index")
indexer_team_2 = StringIndexer(inputCol="Team_2", outputCol="Team_2_index")

# OneHotEncoder for Team_1 and Team_2
encoder_team_1 = OneHotEncoder(inputCol="Team_1_index", outputCol="Team_1_encoded")
encoder_team_2 = OneHotEncoder(inputCol="Team_2_index", outputCol="Team_2_encoded")

# Assemble features
assembler = VectorAssembler(inputCols=["Team_1_encoded", "Team_2_encoded"], outputCol="features")

# StringIndexer for Winner
indexer_winner = StringIndexer(inputCol="Winner", outputCol="label")

# Define the stages of the pipeline
stages = [indexer_team_1, indexer_team_2, encoder_team_1, encoder_team_2, assembler, indexer_winner]

# Create the pipeline
pipeline = Pipeline(stages=stages)

# Fit and transform the data
df_transformed = pipeline.fit(df_teams_all).transform(df_teams_all)

# Select only relevant columns
selected_columns = ["features", "label"]
df_selected = df_transformed.select(selected_columns)

# Show the resulting DataFrame
print("One Hot Encoded dataset")
df_selected.show(truncate=False)

(training_data, testing_data) = df_selected.randomSplit([0.8, 0.2], seed=42)

print("Training data")
training_data.show()

print("Testing data")
testing_data.show()

# Define feature assembler
feature_assembler = VectorAssembler(inputCols=['features'], outputCol='features_vector')
# Define RandomForestClassifier
rf_classifier = RandomForestClassifier(labelCol='label', featuresCol='features_vector', numTrees=500, maxDepth=30, seed=0)

# Create a pipeline
pipeline = Pipeline(stages=[feature_assembler, rf_classifier])

# Fit the model
model = pipeline.fit(training_data)

# Make predictions on the testing data
predictions = model.transform(testing_data)
print("Testing data predictions")
predictions.show()


# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
predictions_training = model.transform(training_data)
eval = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy_training = eval.evaluate(predictions_training)

# Display the accuracy
print("Training Data Accuracy: {:.2%}".format(accuracy_training))
# Display the accuracy
print("Testing Data Accuracy: {:.2%}".format(accuracy))



"""
------------------------------------------------------------------------------------------------
CODE BELOW WAS USED FOR HYPERPARAMETER TUNING USING ParamGridBuilder()
BEST HYPERPARAMETERS FROM ParamGridBuilder() WERE USED FOR TRAINING THE RANDOM FOREST CLASSFIER
-------------------------------------------------------------------------------------------------
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Define feature assembler
feature_assembler = VectorAssembler(inputCols=['features'], outputCol='features_vector')

# Define RandomForestClassifier
rf_classifier = RandomForestClassifier(labelCol='label', featuresCol='features_vector')

# Create a pipeline
pipeline = Pipeline(stages=[feature_assembler, rf_classifier])

# Parameter grid
param_grid = (ParamGridBuilder()
              .addGrid(rf_classifier.numTrees, [500, 600, 750])
              .addGrid(rf_classifier.maxDepth, [10, 20, 30])
              .build())

# Evaluator
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')

# CrossValidator
cross_validator = CrossValidator(estimator=pipeline,
                                 estimatorParamMaps=param_grid,
                                 evaluator=evaluator,
                                 numFolds=2, 
                                 seed=42, parallelism=5)

# Fit the model
cv_model = cross_validator.fit(training_data)


# Print the best set of hyperparameters
best_num_trees = cv_model.bestModel.stages[-1].getOrDefault('numTrees')
best_max_depth = cv_model.bestModel.stages[-1].getOrDefault('maxDepth')

print("Best Hyperparameters:")
print("numTrees: {}".format(best_num_trees))
print("maxDepth: {}".format(best_max_depth))

"""



