# Deep-Q-Learning-agent-for-stock-trading

The aim of this project is to create a model, for predicting the price of stocks from different companies.
The data for this project was pulled via Yahoo Finance API, later pre-procesed and splited into test and train splits, saved into separate files.
For the purpose of the research, I tested a total of 26 different configurations, such that the models are different between them by their values for gamma, epsilon, epsilon min, epsilon decay, learinign rate (value + decay scheduler), weight initiazlition (HeUniform and HeNormal for hidden layers, GlorotUniform and GlorotNormal for output layer).
The result of this research can be seen inside of the Word file in this repository.

The project has 4 sub-folders for key componets:
1. data for saving the csv files
2. core for the functions, which are used inside of training and testing python files
3. model for saving the model for later testing
4. plots for saving the visual from training (profit and reward) and testing (trades)

How can you use this project, with up-to date data:
1.  clone this repository
2.  from the folder data run s&p500.py
3.  from the folder data split-test-train.py
4.  in main.py change the value for the variable tick
5.  run main.py

NOTE: When running the project in this manner, do it in the same day, so that training doesn't finish after midnight and you wouldn't type the model name for testing by hand. 

