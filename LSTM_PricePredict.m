%% Specify Import Options and Load Data
filename = 'AMZNdata.csv';

% Define the expected variable names and types:
opts = delimitedTextImportOptions('Delimiter', ',', 'NumVariables', 8);
opts.VariableNames = {'Date','Open','High','Low','Close','Volume','Dividends','Stock_Splits'};
opts.VariableTypes = {'string','double','double','double','double','double','double','double'};

% Set options to preserve whitespace in the Date column and allow empty fields.
opts = setvaropts(opts, 'Date', 'WhitespaceRule','preserve');
opts = setvaropts(opts, 'Date', 'EmptyFieldRule','auto');

% Read the table using the defined options.
data = readtable(filename, opts);

% Display the variable names for verification
disp('CSV Column Headers:');
disp(data.Properties.VariableNames);

%% Convert Date Column to datetime
% Convert the 'Date' column using the expected format from the Python script.
if ~isdatetime(data.Date)
    try
        % The Python script formats dates as "YYYY-MM-DD HH:MM:SS"
        dates = datetime(data.Date, 'InputFormat', 'yyyy-MM-dd HH:mm:ss', 'Locale', 'en_US');
    catch ME
        error('Failed to convert Date column: %s', ME.message);
    end
else
    dates = data.Date;
end

%% Extract Stock Prices and Clean Data
% We'll use the "High" prices for this example.
if ismember('High', data.Properties.VariableNames)
    prices = data.High;
else
    error('The CSV file does not contain a "High" column.');
end

% Remove any rows with NaN values in prices (and corresponding dates later)
if any(isnan(prices))
    warning('Prices contain NaN values; removing corresponding rows.');
    validIdx = ~isnan(prices);
    prices = prices(validIdx);
    dates  = dates(validIdx);  % Make sure to also update the dates vector
end

% Compute the min and max for normalization
priceMin = min(prices);
priceMax = max(prices);

% Check if the price range is zero to avoid division by zero
if priceMax == priceMin
    warning('Price range is zero; setting normalized prices to zero.');
    pricesNorm = zeros(size(prices));
else
    pricesNorm = (prices - priceMin) / (priceMax - priceMin);
end

% Verify that there are no NaN values now
if any(isnan(pricesNorm))
    error('Normalized prices still contain NaN values.');
end


%% Create Sequences Using a Sliding Window
sequenceLength = 50;  % Use 20 days to predict the next day
numObservations = length(pricesNorm) - sequenceLength;

% Initialize predictors (as a column cell array) and responses (as a column vector)
X = cell(numObservations, 1);
Y = zeros(numObservations, 1);

for i = 1:numObservations
    % Each input is a row vector of normalized prices over 'sequenceLength' days
    X{i} = pricesNorm(i:i+sequenceLength-1)';
    % The target is the price immediately following the sequence
    Y(i) = pricesNorm(i+sequenceLength);
end

%% Partition Data into Training and Testing Sets
% Here, 35% of the observations are used for training, and the rest for testing.
numTrain = floor(0.30 * numObservations);
XTrain = X(1:numTrain);
YTrain = Y(1:numTrain);
XTest  = X(numTrain+1:end);
YTest  = Y(numTrain+1:end);

%% Define the LSTM Network Architecture
layers = [ ...
    sequenceInputLayer(1)
    lstmLayer(100, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer];


% Check for NaNs in normalized prices
if any(isnan(pricesNorm))
    error('Normalized prices contain NaN values.');
end

% Convert cell array X to a numeric array for checking
XAll = cell2mat(X);
if any(isnan(XAll(:)))
    error('Training data contains NaN values.');
end
if any(isnan(Y))
    error('Response data contains NaN values.');
end

%% Set Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 125, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

%% Train the Network
net = trainNetwork(XTrain, YTrain, layers, options);

%% Make Predictions on the Test Set
YPred = predict(net, XTest, 'MiniBatchSize', 1);

% Inverse normalization for both predictions and true values
YPredUnNorm = YPred * (priceMax - priceMin) + priceMin;
YTestUnNorm = YTest * (priceMax - priceMin) + priceMin;

%% Plot the Results
figure;
plot(YTestUnNorm, 'r-', 'LineWidth', 1.5);
hold on;
plot(YPredUnNorm, 'b--', 'LineWidth', 1.5);
legend('Actual Price', 'Predicted Price');
xlabel('Time Step (Test Set)');
ylabel('Stock Price');
title('LSTM Stock Price Prediction');
grid on;


%% %% Evaluate Model Performance
% Calculate error metrics for the test set predictions
errors = YTestUnNorm - YPredUnNorm;
MAE = mean(abs(errors));           % Mean Absolute Error
MSE = mean(errors.^2);             % Mean Squared Error
RMSE = sqrt(MSE);                  % Root Mean Squared Error

% Optionally, compute R-squared
SST = sum((YTestUnNorm - mean(YTestUnNorm)).^2);
SSE = sum((YTestUnNorm - YPredUnNorm).^2);
R2 = 1 - SSE/SST;

% Display the error metrics
fprintf('Mean Absolute Error (MAE): %.4f\n', MAE);
fprintf('Mean Squared Error (MSE): %.4f\n', MSE);
fprintf('Root Mean Squared Error (RMSE): %.4f\n', RMSE);
fprintf('R-squared: %.4f\n', R2);
