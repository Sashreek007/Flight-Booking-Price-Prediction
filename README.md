# Flight-Booking-Price-Prediction
1.
pandas: This library provides data structures and data analysis tools that are perfect for cleaning and analyzing datasets.

numpy: A library for numerical computing in Python. It allows for operations on arrays and matrices, among other numerical tasks.

matplotlib: The primary plotting library in Python. It provides functions for making a wide range of static, animated, and interactive visualizations.

seaborn: Built on top of matplotlib, seaborn provides a higher-level interface for creating attractive visualizations. It integrates well with pandas.

2.
You're reading a CSV file named Flight_Booking.csv into a pandas DataFrame. After reading the CSV file, you drop the column "Unnamed: 0", which can often be an artifact from saving a DataFrame to CSV without setting the index parameter to False. Lastly, you display the first five rows of the DataFrame to get an overview of its structure and content.


3.
df.shape: This returns a tuple representing the dimensions of the DataFrame df. The first value indicates the number of rows, and the second value indicates the number of columns.

df.info(): This method provides a concise summary of the DataFrame, including the number of non-null values in each column, data type of each column, and memory usage. It's especially useful to quickly observe any missing values in the dataset.

df.describe(): This method provides descriptive statistics of the columns in the DataFrame. By default, it includes statistics of only the numeric columns: count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum. If the DataFrame has object type (e.g., strings) or categorical columns, you can include them in the output by passing include='all' as a parameter.

df.isnull().sum()

4.
plt.figure(figsize=(15,5)): This initializes a new figure with a specified size of 15 units in width and 5 units in height.

sns.lineplot(x=df['airline'], y=df['price']): This creates a line plot using the seaborn library. The x-values represent different airlines, while the y-values represent the associated prices. The lineplot function will plot the mean of price for each unique value in airline, and it will also show the 95% confidence interval for the mean by default.

plt.title('Airlines Vs Price', fontsize=15): This sets the title of the plot to "Airlines Vs Price" with a font size of 15.

plt.xlabel('Airline', fontsize=15) and plt.ylabel('Price',fontsize=15): These set the labels for the x and y axes, respectively, both with a font size of 15.

plt.show(): This displays the plot. If you're using a Jupyter notebook or another interactive Python environment, the plot will be shown inline.


5.
plt.figure(figsize=(15,5)): Sets the size of the figure to have a width of 15 units and a height of 5 units.

sns.lineplot(x=df['days_left'], y=df['price'], color='blue'): Creates a line plot using seaborn. The x-values (df['days_left']) represent the number of days left for departure, and the y-values (df['price']) represent the corresponding ticket prices. The line is colored blue as specified by the color argument.

plt.title('Days Left For Departure Vs Ticket Price', fontsize=15): Sets the title of the plot with a font size of 15.

plt.xlabel('Days Left For Departure', fontsize=15) and plt.ylabel('Price',fontsize=15): Set labels for the x and y axes, respectively, both with a font size of 15.

plt.show(): Displays the generated plot.

6.
plt.figure(figsize=(10,5)): Sets the size of the figure to have a width of 10 units and a height of 5 units.

sns.barplot(x='airline', y='price', data=df): Creates a bar plot using seaborn.

The x argument specifies that the x-values will come from the 'airline' column of the DataFrame df.

The y argument specifies that the y-values (heights of the bars) will come from the 'price' column of the DataFrame df.

The data argument specifies the source DataFrame, which in this case is df.


7.
plt.figure(figsize=(10,8)): Sets the size of the figure to have a width of 10 units and a height of 8 units.

sns.barplot(x='class', y='price', data=df, hue='airline'): Creates a bar plot using seaborn.

fig, ax = plt.subplots(1, 2, figsize=(20, 6)): This initializes a figure with two side-by-side subplots (1 row and 2 columns). The size of the entire figure is set to be 20 units wide and 6 units high. The ax object is an array containing two axis objects, one for each subplot.

sns.lineplot(x='days_left', y='price', data=df, hue='source_city', ax=ax[0]): This creates a line plot on the first subplot (ax[0]). The x-values represent the number of days left for departure, the y-values represent ticket prices, and different lines (differentiated by colors) represent different source cities.

sns.lineplot(x='days_left', y='price', data=df, hue='destination_city', ax=ax[1]): Similarly, this creates a line plot on the second subplot (ax[1]). The distinction here is that the lines now represent different destination cities.

plt.show(): Displays the complete visualization.


8.
plt.figure(figsize=(15,23)): Initializes a new figure with a specified size of 15 units in width and 23 units in height.
For each of the following subplots:

plt.subplot(4,2,X): Creates a new subplot in the specified position X on the 4x2 grid.

sns.countplot(...): Creates a count plot using Seaborn. A count plot shows the counts of occurrences of unique values in a specific column.

[plt.title('...')]: Sets the title for the respective subplot.

The specific subplots are:

1st subplot: Shows the frequency distribution of different airlines.
2nd subplot: Displays the frequency distribution of source cities.
3rd subplot: Shows the distribution of departure times.
4th subplot: Visualizes the frequency distribution of the number of stops.
5th subplot: Displays the distribution of arrival times.
6th subplot: Shows the frequency distribution of destination cities.
7th subplot: Visualizes the distribution of different flight classes (e.g., Economy, Business).

plt.show(): This displays the entire visualization with all subplots.

9.
from sklearn.preprocessing import LabelEncoder: This imports the LabelEncoder class from scikit-learn's preprocessing module.

le = LabelEncoder(): This initializes an instance of the LabelEncoder class.

For each of the following columns in the DataFrame:

df['column_name'] = le.fit_transform(df['column_name']):

le.fit_transform(...): This method first fits the encoder with the unique categories in the specified column using the fit method and then transforms these categories into integer labels using the transform method.

The result (integer labels) is then assigned back to the original column in the DataFrame df.

The columns being transformed are:

airline
source_city
departure_time
stops
arrival_time
destination_city
class


df.info(): This method provides a concise summary of the DataFrame, including the number of non-null values in each column, data type of each column, and memory usage. After the label encoding, you should see that the aforementioned columns have been converted to an integer data type.

10.
plt.figure(figsize=(10,5)): Sets up the size of the figure to be 10 units wide and 5 units high.

sns.heatmap(df.corr(), annot=True, cmap='coolwarm'): Creates a heatmap using Seaborn.

df.corr(): Calculates the Pearson correlation coefficients between all pairs of numeric columns in the DataFrame. This method returns a correlation matrix.

annot=True: This ensures that the correlation coefficients are plotted on the heatmap, making it easier to read exact values.

cmap='coolwarm': Specifies the colormap for the heatmap. 'coolwarm' is a diverging colormap, where cooler colors (blues) represent negative correlations, warmer colors (reds) represent positive correlations, and neutrals (whites) represent little to no correlation.

plt.show(): Displays the heatmap.

11.
The code first checks if the columns 'stops' and 'flight' exist in the DataFrame. If they do, they are dropped.

Then, the variance_inflation_factor function from statsmodels is imported. This function calculates the VIF for each predictor variable.

An empty list col_list is created. The for loop populates this list with column names from the DataFrame that are numeric (excluding 'object' type) and aren't the 'price' column.

X is created as a subset of df containing only the columns in col_list.

An empty DataFrame vif_data is created to store the VIF values for each feature.

The VIF is calculated for each column in X using a list comprehension, and the results are stored in vif_data under the 'VIF' column.

Finally, the VIF values for each feature are printed.

12.
x = df.drop(columns=['price']): This creates a new DataFrame x by dropping the 'price' column from df. This means x contains all the features (predictor variables) you intend to use for prediction.

y = df['price']: This creates a Series y which contains the target variable (what you want to predict) - in this case, the 'price'.

from sklearn.model_selection import train_test_split: This imports the train_test_split function from scikit-learn, which is used to split data arrays or matrices into random train and test subsets.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42): This splits the data into training and testing sets.

x and y are the features and target variable, respectively.

test_size=0.2 means that 20% of the data will be reserved for testing, and the remaining 80% will be used for training.

random_state=42 is used for reproducibility. Setting a random_state ensures that the splits generate the same train/test split each time the code is run.

After executing this code, you'll have four sets:

x_train and y_train: The features and target for the training set.

x_test and y_test: The features and target for the testing set.


13.
from sklearn.linear_model import LinearRegression: This imports the LinearRegression class from scikit-learn's linear_model module.

lr = LinearRegression(): Initializes an instance of the LinearRegression class.

lr.fit(x_train, y_train): This fits (or trains) the linear regression model using the training data x_train and the corresponding target values y_train.

y_pred = lr.predict(x_test): Once the model is trained, this line predicts the target values for the test data x_test. The predictions are stored in the y_pred array.

difference = pd.DataFrame(np.c_[y_test, y_pred], columns=['Actual_value', 'Predicted_value']): This creates a new DataFrame named difference which consists of two columns:

Actual_value contains the true target values from y_test.

Predicted_value contains the predicted values from y_pred.

The function np.c_[y_test, y_pred] is used to concatenate the two arrays column-wise.

14.
This code evaluates the performance of the linear regression model using several metrics. These metrics help determine how well the model's predictions match the actual values:

R-squared (Coefficient of Determination): This statistic measures the proportion of the variance in the dependent variable that's predictable from the independent variables.

r2_score = r2_score(y_test,y_pred): Computes the R-squared value.
Mean Absolute Error (MAE): This is the average of the absolute differences between the predicted and actual values. It gives an idea of the magnitude of the error but doesn't indicate its direction.

mean_abs_error = metrics.mean_absolute_error(y_test,y_pred): Computes the MAE.
Mean Absolute Percentage Error (MAPE): This metric represents the average absolute percentage difference between the observed actual outcomes and the predictions.

mean_absolute_percentage_error = mean_absolute_percentage_error(y_test,y_pred): Computes the MAPE.
Mean Squared Error (MSE): This is the average of the squared differences between the predicted and actual values. It emphasizes larger errors over smaller ones.

mean_sq_error= metrics.mean_squared_error(y_test,y_pred): Computes the MSE.
Root Mean Squared Error (RMSE): This is the square root of the MSE. It's useful because its units are the same as the target variable, and it gives an idea of the magnitude of the error.

root_mean_sq_error= np.sqrt(metrics.mean_squared_error(y_test,y_pred)): Computes the RMSE.

15.
sns.distplot(y_test, label="Actual"): This plots a histogram of the actual values (y_test) with a kernel density estimate (KDE) overlay. The label="Actual" part assigns a label for this plot, which will be used in the legend.

sns.distplot(y_pred, label="Predicted"): This plots a histogram of the predicted values (y_pred) with a KDE overlay. Similarly, label="Predicted" assigns a label for this plot.

plt.legend(): Displays the legend on the plot, which will differentiate between the "Actual" and "Predicted" distributions.

16.
from sklearn.tree import DecisionTreeRegressor: This imports the DecisionTreeRegressor class from scikit-learn's tree module.

dt = DecisionTreeRegressor(): Initializes an instance of the DecisionTreeRegressor class. This creates a new decision tree regressor.

dt.fit(x_train, y_train): This fits (or trains) the decision tree model using the training data x_train and the corresponding target values y_train.

y_pred = dt.predict(x_test): Once the model is trained, this line predicts the target values for the test data x_test. The predictions are stored in the y_pred array.

Decision trees are a type of model that make decisions based on asking a series of questions. For regression, they predict the target variable based on features in the data, but instead of outputting a class label, they output a continuous value. Decision trees can sometimes overfit to the training data, especially if they are very deep, so it's essential to evaluate their performance on test data and possibly consider techniques like pruning or using ensemble methods like Random Forests for more robust predictions.
17.sns.distplot(y_test, label="Actual"): This will plot the distribution of the actual values from the test set.

sns.distplot(y_pred, label="Predicted"): This will plot the distribution of the predicted values from the decision tree model.

plt.legend(): This displays a legend on the plot to differentiate between the "Actual" and "Predicted" distributions.

18.
from sklearn.ensemble import RandomForestRegressor: Imports the RandomForestRegressor class from scikit-learn's ensemble module.

rfr = RandomForestRegressor(): Initializes an instance of the RandomForestRegressor class. This creates a new random forest regressor with default parameters.

rfr.fit(x_train, y_train): Fits (or trains) the random forest regressor using the training data x_train and the corresponding target values y_train.

y_pred = rfr.predict(x_test): Predicts the target values for the test data x_test using the trained model. The predictions are stored in the y_pred array.

19.
R-squared (Coefficient of Determination):

r2_score = r2_score(y_test,y_pred): This computes the R-squared value. It represents the proportion of the variance in the dependent variable that's predictable from the independent variables.
Mean Absolute Error (MAE):

mean_abs_error = metrics.mean_absolute_error(y_test,y_pred): This computes the MAE. It gives an idea of how off the predictions are on average.
Mean Absolute Percentage Error (MAPE):

mean_absolute_percentage_error = mean_absolute_percentage_error(y_test,y_pred): This computes the MAPE, which represents the average absolute percentage difference between the observed actual outcomes and the predictions.
Mean Squared Error (MSE):

mean_sq_error = metrics.mean_squared_error(y_test,y_pred): This computes the MSE. It emphasizes the impact of larger errors.
Root Mean Squared Error (RMSE):

root_mean_sq_error = np.sqrt(metrics.mean_squared_error(y_test,y_pred)): This computes the RMSE. It is interpreted in the same units as the response variable and gives an idea of the magnitude of the error.

20.
sns.distplot(y_test, label="Actual"): This plots the distribution of the actual test values.

sns.distplot(y_pred, label="Predicted"): This plots the distribution of the predicted values from the Random Forest model.

plt.legend(): This displays a legend on the plot to differentiate between the "Actual" and "Predicted" distributions.


