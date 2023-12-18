**Folder structure:**
- Full_batting_data/
  - player1_data.csv
  - player2_data.csv
  - ...
- Full_bowling_data/
  - player1_data.csv
  - player2_data.csv
  - ...
- playerPrediction.py
- README.md
- plotBatting.py
- plotBowling.py
- rf_classifier.py
- results.csv
- World_cup_2023.csv
- Icc_ranking.csv

**Requirements**
- Python 3.x
- PySpark
- Pandas
- Matplotlib



1. **plotBatting.py**

    This Python script analyzes batting data for cricket players stored in CSV files within the "Full_batting_data" folder. It uses PySpark for data processing and matplotlib for visualization. The script reads each CSV file, processes the data, calculates the average strike rate per year, and plots a bar chart for each player.

    **To run the file,**

    "Full_batting_data" is required (should be in the same directory as plotBatting.py)
    Use command python3 plotBatting.py

2. **plotBowling.py**

    This Python script analyzes bowling data for cricket players stored in CSV files within the "Full_bowling_data" folder. It uses PySpark for data processing and matplotlib for visualization. The script reads each CSV file, processes the data, calculates the average economy rate per year, and plots a bar chart for each player.

    **To run the file,**

    "Full_bowling_data" is required (should be in the same directory as plotBowling.py)
    Use command python3 plotBowling.py

3. **playerPrediction.py**

    This Python script predicts the future performance of cricket players using PySpark for data processing and linear regression for modeling. The script analyzes both batting and bowling data and predicts future performance based on historical trends. *Predictions are made for the year 2023, and the results are compared with actual values.*

    **To run the file,**

    "Full_bowling_data" and "Full_batting_data" is required (should be in the same directory as playerPrediction.py)
    Use command python3 playerPrediction.py

4. **rf_classifier.py**

    This Python script uses PySpark to analyze cricket match data, perform feature engineering, and predict match outcomes using a Random Forest Classifier. The analysis includes processing data related to World Cup matches, team rankings, and match results. *The predictions are based on features such as the teams involved and historical match outcomes.*

    **To run the file,**

    results.csv,World_cup_2023.csv,Icc_ranking.csv is required (should be in the same directory as rf_classifier.py)
    Use command python3 rf_classifier.py


5. **Helper files**

   getBattingData.py, getBowlingData.py, getPlayerIds.py are helper files which is used to get the data in "Full_bowling_data" and "Full_batting_data" from https://www.espncricinfo.com/ website.


