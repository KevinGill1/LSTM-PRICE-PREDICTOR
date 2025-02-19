%% Load and Prepare Data
filename = 'NVDAdata.csv';

% Define the expected variable names and types:
opts = delimitedTextImportOptions('Delimiter', ',', 'NumVariables', 8);
opts.VariableNames = {'Date','Open','High','Low','Close','Volume','Dividends','Stock_Splits'};
opts.VariableTypes = {'string','double','double','double','double','double','double','double'};
data = readtable(filename, opts);

% Convert Date column to datetime if needed
if ~isdatetime(data.Date)
    try
        dates = datetime(data.Date, 'InputFormat', 'yyyy-MM-dd HH:mm:ss', 'Locale', 'en_US');
    catch ME
        error('Failed to convert Date column: %s', ME.message);
    end
else
    dates = data.Date;
end

% Remove rows with missing data in key predictors or response
validRows = ~isnan(data.Open) & ~isnan(data.Low) & ~isnan(data.Close) & ~isnan(data.Volume) & ~isnan(data.High);
data = data(validRows, :);
dates = dates(validRows);

%% Define Predictor Variables and Response
% Use Open, Low, Close, and Volume as predictors
X = data{:, {'Open','Low','Close','Volume'}};
% Response variable: High price
Y = data.High;

%% Partition Data into Training and Testing Sets
% Use 70% for training and 30% for testing.
cv = cvpartition(size(X,1), 'HoldOut', 0.25);
idxTrain = training(cv);
idxTest  = test(cv);

XTrain = X(idxTrain, :);
YTrain = Y(idxTrain);
XTest  = X(idxTest, :);
YTest  = Y(idxTest);

%% Create a Table for Linear Regression
% Build a table from the training data.
trainTable = array2table(XTrain, 'VariableNames', {'Open','Low','Close','Volume'});
trainTable.High = YTrain;

%% Train a Linear Regression Model using fitlm
linModel = fitlm(trainTable, 'High ~ Open + Low + Close + Volume');

%% Make Predictions on the Test Set
% Create a test table using the same variable names.
testTable = array2table(XTest, 'VariableNames', {'Open','Low','Close','Volume'});
YPred = predict(linModel, testTable);

%% Evaluate Model Performance
errors = YTest - YPred;
MAE = mean(abs(errors));           % Mean Absolute Error
MSE = mean(errors.^2);             % Mean Squared Error
RMSE = sqrt(MSE);                  % Root Mean Squared Error

% Compute R-squared (Coefficient of Determination)
SST = sum((YTest - mean(YTest)).^2);
SSE = sum(errors.^2);
R2 = 1 - SSE/SST;

fprintf('Linear Regression Model Performance:\n');
fprintf('Mean Absolute Error (MAE): %.4f\n', MAE);
fprintf('Mean Squared Error (MSE): %.4f\n', MSE);
fprintf('Root Mean Squared Error (RMSE): %.4f\n', RMSE);
fprintf('R-squared: %.4f\n', R2);

%% Plot Actual vs. Predicted High Prices
figure;
plot(YTest, 'r-', 'LineWidth', 1.5);
hold on;
plot(YPred, 'b--', 'LineWidth', 1.5);
legend('Actual High Price', 'Predicted High Price');
xlabel('Test Sample Index');
ylabel('High Price');
title('Linear Regression: Actual vs. Predicted High Prices');
grid on;
