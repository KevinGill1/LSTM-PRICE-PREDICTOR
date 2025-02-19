%% Specify Import Options and Load Data
filename = 'NVDAdata.csv';
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

if ismember('Volume', data.Properties.VariableNames)
    volume = data.Volume;
else
    error('The CSV file does not contain a "Volume" column.');
end


% Remove rows with NaN values in either prices or volume (and update dates accordingly)
nanIdx = isnan(prices) | isnan(volume);
if any(nanIdx)
    warning('Prices or Volume contain NaN values; removing corresponding rows.');
    prices = prices(~nanIdx);
    volume = volume(~nanIdx);
    dates  = dates(~nanIdx);
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

% Normalize Volume separately
volMin = min(volume);
volMax = max(volume);
if volMax == volMin
    warning('Volume range is zero; setting normalized volume to zero.');
    volumeNorm = zeros(size(volume));
else
    volumeNorm = (volume - volMin) / (volMax - volMin);
end

if any(isnan(pricesNorm)) || any(isnan(volumeNorm))
    error('Normalized data still contain NaN values.');
end


% % Verify that there are no NaN values now
% if any(isnan(pricesNorm))
%     error('Normalized prices still contain NaN values.');
% end


%% Create Sequences Using a Sliding Window
sequenceLength = 50;  % Use 20 days to predict the next day (for high price)
numObservations = length(pricesNorm) - sequenceLength;

% Initialize predictors (each observation is a matrix with 2 rows: [high; volume]) 
% and responses (the target is the high price of the day following the sequence)
X = cell(numObservations, 1);
Y = zeros(numObservations, 1);

for i = 1:numObservations
    % Create a 2 x sequenceLength matrix:
    % Row 1: normalized high prices, Row 2: normalized volume
    X{i} = [pricesNorm(i:i+sequenceLength-1)'; volumeNorm(i:i+sequenceLength-1)'];
    % The target is the normalized high price immediately following the sequence
    Y(i) = pricesNorm(i+sequenceLength);
end

%% Partition Data into Training and Testing Sets
numTrain = floor(0.3 * numObservations);
XTrain = X(1:numTrain);
YTrain = Y(1:numTrain);
XTest  = X(numTrain+1:end);
YTest  = Y(numTrain+1:end);

%% Define the LSTM Network Architecture with Two Input Features
layers = [ ...
    sequenceInputLayer(2)  % Now we have 2 features: high price and volume
    lstmLayer(100, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer];

%% Set Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.001, ... % Lowered learning rate for stability
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 125, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

%% Train the Network
net = trainNetwork(XTrain, YTrain, layers, options);

%% Make Predictions on the Test Set
YPred = predict(net, XTest, 'MiniBatchSize', 1);

% Inverse normalization for the high price predictions and true values
YPredUnNorm = YPred * (priceMax - priceMin) + priceMin;
YTestUnNorm = YTest * (priceMax - priceMin) + priceMin;

%% Plot the Results
figure;
plot(YTestUnNorm, 'r-', 'LineWidth', 1.5);
hold on;
plot(YPredUnNorm, 'b--', 'LineWidth', 1.5);
legend('Actual High Price', 'Predicted High Price');
xlabel('Time Step (Test Set)');
ylabel('Stock Price');
title('LSTM Stock Price Prediction Using High Price and Volume');
grid on;

%% Evaluate Model Performance
errors = YTestUnNorm - YPredUnNorm;
MAE = mean(abs(errors));
MSE = mean(errors.^2);
RMSE = sqrt(MSE);
SST = sum((YTestUnNorm - mean(YTestUnNorm)).^2);
SSE = sum((YTestUnNorm - YPredUnNorm).^2);
R2 = 1 - SSE/SST;

fprintf('Mean Absolute Error (MAE): %.4f\n', MAE);
fprintf('Mean Squared Error (MSE): %.4f\n', MSE);
fprintf('Root Mean Squared Error (RMSE): %.4f\n', RMSE);
fprintf('R-squared: %.4f\n', R2);