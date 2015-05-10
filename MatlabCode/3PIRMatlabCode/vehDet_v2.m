% class vehDet
% Raphael, 
% Detect vehicles using logistic regression
% input training data => model; call vehTrain
% input test data & model => detect_veh; call vehTest


classdef vehDet < handle
    
    properties
        %========================================
        % keep a copy of all the input data
        
        % the raw data of all sensors 
        % [MATLAB_time, PIR1, PIR2, PIR3], num_meas x 4
        % Assume the test portion has been removed; in MATLAB time format
        raw_data; 
        
        % [start_time, end_time, velocity]: num_training_samples x 3
        % start time and end time of each vehicle detection
        % time in Matlab format (7.3592e+05)
        train_label;
        
        % test data [start_time, end_time]: num_test x 2 
        % each test data contains only one vehicle detection
        test_label;
        
        
        % training data
        % this is the Hamming window averaged measurement for a sensor for
        % a time period
        train_data;
        
        %test_data;
        
        train_label_value;
        %========================================
        % internal data for this class
        
        % the logistic regression model
        % essentially a vector containing coefficients
        model;
        
        T;
        s;
        
    end
    
    
    methods
        %===============================================================
        % initialize object properties with all the raw data
        function obj = vehDet(raw_data, T, s)
            
            obj.T = T;
            obj.s = s;
            
            obj.raw_data = raw_data;
            obj.train_label = [];
            %obj.test_label = [];
            obj.train_data = [];
            %obj.test_data = [];
            obj.train_label_value = [];
                       
        end
        
        
        
        %===============================================================
        % average a signal using a Hamming window
        % input: signal, window size T, step size s
        % output: averaged value of signal
        function avgVal = average_signal(obj, signal)
            numWindows = floor(size(signal, 1)/obj.T);
            
            avgVal = zeros(numWindows, 1);
            curser = 1;
            
            for i = 1:obj.s:obj.T*(numWindows-1)
                avgVal(curser) = (hamming(obj.T)'*signal(i:(i+obj.T-1)))/sum(hamming(obj.T));
                curser = curser + 1;
            end
            
        end
            
        
        %===============================================================
        % train model
        % input: train_label: [start_time, end_time]
        % output: linear regression model
        function model = detectionTrain(obj, train_label)
            
            obj.train_label = train_label;
                   
            obj.train_data = [obj.average_signal(obj.raw_data(:,2)), ...
                              obj.average_signal(obj.raw_data(:,3)), ...
                              obj.average_signal(obj.raw_data(:,4))];
            
            num_points = size(obj.train_label, 1);
            
            index = (zeros(size(obj.raw_data, 1), 1) == 1);
            
            for i = 1:num_points
                temp_index = (obj.raw_data(:, 1) >= obj.train_label(i, 1)) ...
                    & (obj.raw_data(:,1) <= obj.train_label(i, 2));
                
                index = index | temp_index;
            end
            
            obj.train_label_value = (obj.average_signal(index)>0.5);
                          
            % use logistic regression to fit the model
            model = mnrfit(obj.train_data, obj.train_label_value+1);
            
            % save a copy as property
            obj.model = model;
           
        end
        
        
        %===============================================================
        % input: test_label [start_time, end_time]; 
        % ouput num_counts
        function detected_veh = vehTest(obj,test_interval)
            
            obj.test_label = test_interval;
            
            %detected_veh = zeros(size(test_label,1),1);
            
            all_data = obj.extractMatrix(obj.raw_data,...
                            test_interval(1), test_interval(2));
            
            test_data = [obj.average_signal(all_data(:,2)), ...
                         obj.average_signal(all_data(:,3)), ...
                         obj.average_signal(all_data(:,4))];
            
            
            detected_veh = mnrval(obj.model, test_data);
            detected_veh = (detected_veh >= 0.5);

            
        end
        
        
        
        %===============================================================
        % extract a submatrix between start_time, end_time in the matrix
        % assume the matrix is sorted by the time
        function submatrix = extractMatrix(obj, matrix, start_time, end_time)
            
            index_greater = matrix(:,1) >= start_time;
            index_smaller = matrix(:,1) <= end_time;
            index_between = index_greater & index_smaller;
            
            submatrix = matrix(index_between,:);
            
        end
        
        
    end
    
end













