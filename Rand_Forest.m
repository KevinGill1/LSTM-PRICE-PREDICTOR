%% Load and Prepare Data
filename = 'NVDAdata.csv';

% Define the expected variable names and types
opts = delimitedTextImportOptions('Delimiter', ',', 'NumVariables', 8);
opts.VariableNames = {'Date','Open','High','Low','Close','Volume','Dividends','Stock_Splits'};
opts.VariableTypes = {'string','double','double','double','double','double','double','double'};
data = readtable(filename, opts);

% (Optional) Convert Date column to datetime if needed
if ~isdatetime(data.Date)
    try
        dates = datetime(data.Date, 'InputFormat', 'yyyy-MM-dd HH:mm:ss', 'Locale', 'en_US');
    catch ME
        error('Failed to convert Date column: %s', ME.message);
    end
else
    dates = data.Date;
end

% Remove any rows with missing data in key predictors or response
validRows = ~isnan(data.Open) & ~isnan(data.Low) & ~isnan(data.Close) & ~isnan(data.Volume) & ~isnan(data.High);
data = data(validRows, :);
dates = dates(validRows);

%% Define Predictor Variables and Response
% We will use Open, Low, Close, and Volume as predictors
X = data{:, {'Open','Low','Close','Volume'}};
% Response variable: High price
Y = data.High;

%% Partition Data into Training and Testing Sets
% Let's use 55% of the data for training and 45% for testing.
cv = cvpartition(size(X,1), 'HoldOut', 0.45);
idxTrain = training(cv);
idxTest  = test(cv);

XTrain = X(idxTrain, :);
YTrain = Y(idxTrain);
XTest  = X(idxTest, :);
YTest  = Y(idxTest);

%% Train a Random Forest Regressor using TreeBagger
numTrees = 150;  % You can experiment with more trees
rfModel = TreeBagger(numTrees, XTrain, YTrain, 'Method', 'regression', 'OOBPrediction', 'on');

% (Optional) View out-of-bag error plot
figure;
oobErrorBaggedEnsemble = oobError(rfModel);
plot(oobErrorBaggedEnsemble);
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Mean Squared Error');
title('OOB Error Estimate');

%% Make Predictions on the Test Set
YPred = predict(rfModel, XTest);

% Check if predictions are a cell array; if so, convert to numeric.
if iscell(YPred)
    YPred = cell2mat(YPred);
end


%% Evaluate Model Performance
% Compute error metrics
errors = YTest - YPred;
MAE = mean(abs(errors));           % Mean Absolute Error
MSE = mean(errors.^2);             % Mean Squared Error
RMSE = sqrt(MSE);                  % Root Mean Squared Error

% Compute R-squared (Coefficient of Determination)
SST = sum((YTest - mean(YTest)).^2);
SSE = sum(errors.^2);
R2 = 1 - SSE/SST;

fprintf('Random Forest Model Performance:\n');
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
title('Random Forest: Actual vs. Predicted High Prices');
grid on;
