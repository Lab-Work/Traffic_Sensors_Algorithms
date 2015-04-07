% class vehDet
% Raphael, 
% Detect vehicles using logistic regression
% input training data => model; call vehTrain
% input test data & model => detect_veh; call vehTest


% version v3, 
% added gmm model


classdef vehDet_v3 < handle
    
    properties
        %========================================
        % keep a copy of all the input data
        
        % the raw data of all sensors 
        % [MATLAB_time, PIR1, PIR2, PIR3], num_meas x 4
        % Assume the test portion has been removed; in MATLAB time format
        %raw_data; 
        
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
        
        %train_label_value;
        %========================================
        % internal data for this class
        
        % the logistic regression model
        % essentially a vector containing coefficients
        model;
        
        T;
        s;
        
        
        % for v3
        % use model
        use_model; 
        % 'gmm' gaussian mixture model
        % 'logit' Logistic regression
        
        
        % prior for gmm
        use_prior;  % 'yes', 'no'
        prior_veh;
        
    end
    
    
    methods
        %===============================================================
        % initialize object properties with all the raw data
        function obj = vehDet_v3(T, s, use_model, use_prior)
            
            obj.T = T;
            obj.s = s;
            obj.use_model = use_model;
            
            %obj.raw_data = raw_data;
            obj.train_label = [];
            %obj.test_label = [];
            obj.train_data = [];
            %obj.test_data = [];
            %obj.train_label_value = [];
            
            obj.use_prior = use_prior;
                       
        end
        
        
        
        %===============================================================
        % average a signal using a Hamming window
        % input: signal, window size T, step size s
        % output: averaged value of signal
        function avgVal = average_signal(obj, signal)
            
            % first apprach
%             numWindows = floor( (size(signal, 1)-obj.T)/obj.s)+1;
%             
%             avgVal = zeros(numWindows, 1);
%             curser = 1;
%             
%             % i = 1:obj.s:obj.T*(numWindows-1)
%             for i = 1:obj.s:size(signal,1)-obj.T
%                 avgVal(curser) = (hamming(obj.T)'*signal(i:(i+obj.T-1)))/sum(hamming(obj.T));
%                 curser = curser + 1;
%             end

              
            % second approach
            numWindows = floor( (size(signal, 1)-obj.T)/obj.s)+1;
            
            chopped_signal = zeros(numWindows, obj.T);
            
            for i = 1:numWindows
                
                chopped_signal(i,:) = signal( (i-1)*obj.s+1 : (i-1)*obj.s + obj.T )';
                
            end
            
            % apply window
            avgVal = chopped_signal*hamming(obj.T)/sum(hamming(obj.T));


            
        end
        
        
        %===============================================================
        % train model
        % input: train_data: [time, PIR1, PIR2, PIR3]; train_label: [start_time, end_time]
        % output: linear regression model
        function model = detectionTrain(obj, train_data, train_label)
            

            
            %obj.train_label = train_label;
                   
            windowed_train_data = [
                              obj.average_signal(train_data(:,2)), ...
                              obj.average_signal(train_data(:,3)), ...
                              obj.average_signal(train_data(:,4))
                              ];
            
            
            index = (zeros(size(train_data, 1), 1) == 1);
         
            for i = 1:size(train_label, 1)
                temp_index = (train_data(:, 1) >= train_label(i, 1)) ...
                    & (train_data(:,1) <= train_label(i, 2));
                
                index = index | temp_index;
            end
            
            train_label_value = (obj.average_signal(index)>0.5);
                          
            % use logistic regression to fit the model
            if strcmp(obj.use_model, 'logit')
                model = mnrfit(windowed_train_data, train_label_value+1);
            elseif strcmp(obj.use_model, 'gmm')
                
                obj.prior_veh = sum(train_label_value)/length(train_label_value);
                
                % train non vehicle data
                Str = 'nonVeh';
                nonVeh_train_data = windowed_train_data( train_label_value==0, : );
                model.(Str).mu = mean(nonVeh_train_data,1);
                model.(Str).sigma = cov(nonVeh_train_data);
                
                % train vehicle data
                Str = 'veh';
                veh_train_data = windowed_train_data( train_label_value==1, : );
                model.(Str).mu = mean(veh_train_data,1);
                model.(Str).sigma = cov(veh_train_data);
            end
            
            % save a copy as property
            obj.model = model;
           
        end
        
        
        %===============================================================
        % input: test_label: [start_time, end_time]; test_data: [time, PIR1,
        % PIR2, PIR3].
        % ouput num_counts
        function detected_veh = vehTest(obj, raw_test_data)
            
            
            %numWindows = floor(size(raw_test_data, 1)/obj.T);
            numWindows = floor( (size(raw_test_data, 1)-obj.T)/obj.s)+1;
            
            start_times = zeros(numWindows, 1);
            end_times = zeros(numWindows, 1);
            
            % save window times
            for i = 1:numWindows
                start_times(i) = raw_test_data( (i-1)*obj.s+1 ,1);
                end_times(i) = raw_test_data( (i-1)*obj.s+obj.T ,1);
            end
            
            test_data = [
                         obj.average_signal(raw_test_data(:,2)), ...
                         obj.average_signal(raw_test_data(:,3)), ...
                         obj.average_signal(raw_test_data(:,4))
                         ];
            
            if strcmp(obj.use_model, 'logit')
                veh_prob = mnrval(obj.model, test_data);
                applied_labels = (veh_prob <= 0.5);
            elseif strcmp(obj.use_model, 'gmm')
                
                % probability of not a vehicle
                Str = 'nonVeh';
                p_nonVeh = mvnpdf(test_data,...
                    obj.model.(Str).mu,obj.model.(Str).sigma);

                % probability of a vehicle
                Str = 'veh';
                p_veh = mvnpdf(test_data,...
                    obj.model.(Str).mu,obj.model.(Str).sigma);
                
                if strcmp(obj.use_prior,'yes')
                    applied_labels = ( (1-obj.prior_veh)*p_nonVeh <= obj.prior_veh*p_veh );
                elseif strcmp(obj.use_prior,'no')
                    applied_labels = ( p_nonVeh <= p_veh);
                else
                    error('invalid use_prior argument');
                end
            else
                error('invalid useModel argument');
            end
            
            
            detected_veh = findDetTimes([start_times, end_times, applied_labels]);
            

            
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













