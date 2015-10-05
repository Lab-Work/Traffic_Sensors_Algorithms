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
        %test_label;
        
        
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
 
        
    end
    
    
    methods
        %===============================================================
        % initialize object properties with all the raw data
        function obj = vehDet(raw_data)
            
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
        function avgVal = average_signal(signal, T, s)
            numWindows = floor(size(signal, 1)/T);
            
            avgVal = zeros(numWindows, 1);
            curser = 1;
            
            for i = 1:s:T*numWindows
                avgVal(curser) = (hamming(T)'*signal(i:(i+T)))/sum(hamming(T));
                curser = curser + 1;
            end
            
        end
            
        
        %===============================================================
        % train model
        % input: train_label: [start_time, end_time]
        % output: linear regression model
        function model = detectionTrain(obj, train_label)
            
            obj.train_label = train_label;
                   
            obj.train_data = [average_signal(obj.raw_data(:,2)), ...
                              average_signal(obj.raw_data(:,3)), ...
                              average_signal(obj.raw_data(:,4))];
            
            num_points = size(obj.raw_data, 1);
            
            index = (zeros(num_points, 1) == 1);
            
            for i = 1:num_points
                temp_index = (obj.raw_data(:, 1) >= obj.train_label(i, 1)) ...
                    && (obj.raw_data(:,1) <= obj.train_label(i, 2));
                
                index = index || temp_index;
            end
            
            obj.train_label_value = (average_signal(index, 50, 25)>0.5);
                          
            % use logistic regression to fit the model
            model = mnrfit(obj.train_data, obj.train_label_value);
            
            % save a copy as property
            obj.model = model;
           
        end
        
        
        %===============================================================
        % input: test_label [start_time, end_time]; 
        % ouput num_counts
        function detected_veh = vehTest(obj,test_label)
            
            obj.test_label = test_label;
            
            %detected_veh = zeros(size(test_label,1),1);
            
            
            
            test_data = [average_signal(obj.raw_data(test_label(1):test_label(2),2)), ...
                              average_signal(obj.raw_data(test_label(1):test_label(2),3)), ...
                              average_signal(obj.raw_data(test_label(1):test_label(2),4))];
            
            
            detected_veh = mnrval(obj.model, test_data);
            detected_veh = (detected_veh >= 0.5);

            
        end
        
        
        
    end
    
end













