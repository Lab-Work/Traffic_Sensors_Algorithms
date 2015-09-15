% class speedEst
% Yanning, 
% Estimate the vehicle speed using LR using time shifts
% input training data => model; call speedTrain
% input test data & model => est_speed; call speedTest


classdef speedEst < handle
    
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
        
        
        %========================================
        % internal data for this class
        
        % coefficients of the regression model
        % essentially a vector containing coefficients
        model;
        
        % coefficient of the linear regression model using t, w, h
        model_ad;
        
        % time laps [laps12, laps13, laps23] in seconds
        % each row is a sample
        time_shifts;
        
        % constant to reduce potential numerical error
        ratio;
        
        % save the cleaned training label
        cleaned_train_label;
        
        
        % different convolution
        useConv;  % a string
        % 'wrapConv': wrapping data
        % 'zeroPadConv': zero padding data
        
        % different ways of finding shifts
        useShift;   % a string
        % 'row': row shifts
        % 'time': time shifts
        
        % different ways of finding the peak
        usePeak;    % a string
        % 'max': the maximal point of convolution as the shift
        % 'halfThreshold': center of mass of cutted convoluiton by 1/2
        % threshold
        
        % different models, where 1/t or t
        useModel;   % a string
        % 't'   % nonlinear model v = a*t
        % '1/t' % linear model v=a/t
        
        
        % estimated speed
        est_speed;
        
        
        % add more information for the linear regression
        % The width of the peaks; and the height of the peaks
        peakWidth;
        peakHeight;
        
        
    end
    
    
    methods
        %===============================================================
        % initialize object properties with all the raw data
        function obj = speedEst(raw_data)
            
            obj.raw_data = raw_data;
            obj.train_label = [];
            obj.test_label = [];
            
            obj.time_shifts = [];
            obj.ratio = 1;    % now use timeshifts in seconds        
            
            obj.peakWidth = [];
            obj.peakHeight = [];
            
        end
        
        
        %===============================================================
        % train model
        % input: train_label: [start_time, end_time, velocity]
        % output: linear regression model using t12, t23, t13
        function model = speedTrain(obj, train_label,...
                useConv, useShift, usePeak, useModel)
            
            obj.train_label = train_label;
            obj.time_shifts = zeros(0,3);   % reset
            
            obj.useConv = useConv;
            obj.useShift = useShift;
            obj.usePeak = usePeak;
            obj.useModel = useModel;
            
            len_sample = size(obj.train_label,1);
            
            % copmute the time shifts from training data
            for i=1:len_sample
                
                sample = obj.extractMatrix(obj.raw_data,...
                    obj.train_label(i,1), obj.train_label(i,2));
                
                if i == 11
                   1; 
                end
                
                % find time shifts
                if strcmp(obj.useShift,'row')
                    tmp_shifts = obj.rowShifts(sample);
                elseif strcmp(obj.useShift, 'time')
                    tmp_shifts = obj.timeShifts(sample);
                else
                    error('invalid useShift argument');
                end
                
                if ~isempty(tmp_shifts)
                    obj.time_shifts(i,:) = tmp_shifts;
                else
                    str = sprintf('train label row: %d',i);
                    disp(str);
                end
            end
            
            % use linear regression, train the coefficients
            % model = 1 + 1/shift12 + 1/shift13 + 1/shift23
            if strcmp(obj.useModel, '1/t')
                model = fitlm(obj.ratio./obj.time_shifts, obj.train_label(:,3));
            elseif strcmp(obj.useModel, 't')
                model = fitlm(obj.time_shifts, obj.train_label(:,3));
            end
            % save a copy as property
            obj.model = model;
            
            obj.cleaned_train_label = train_label;
           
        end
        
        %===============================================================
        % train a more informative regression model
        % input: train_label: [start_time, end_time, velocity]
        % output: linear regression model
        % model: v = linearFunction(t12, t23, t13, w1, w2, w3, h1, h2, h3)
        function model_ad = speedTrain_advanced(obj, train_label)
            
            obj.train_label = train_label;
            obj.time_shifts = zeros(0,3);   % reset
            
            % According to our previous result, we use the following
            % default parameters:
            
            obj.useConv = 'zeroPadConv';
            obj.useShift = 'row';
            obj.usePeak = 'halfThreshold';
            obj.useModel = '1/t';
            
            % every time we train a new model, reset the values
            obj.peakWidth = [];
            obj.peakHeight = [];
         
            
            len_sample = size(obj.train_label,1);
            
            % copmute the time shifts from training data
            for i=1:len_sample
                
                sample = obj.extractMatrix(obj.raw_data,...
                    obj.train_label(i,1), obj.train_label(i,2));
                
                if i == 11
                   1; 
                end
                
                % find time shifts
                if strcmp(obj.useShift,'row')
                    tmp_shifts = obj.rowShifts(sample);
                elseif strcmp(obj.useShift, 'time')
                    tmp_shifts = obj.timeShifts(sample);
                else
                    error('invalid useShift argument');
                end
                
                if ~isempty(tmp_shifts)
                    obj.time_shifts(i,:) = tmp_shifts;
                else
                    str = sprintf('train label row: %d',i);
                    disp(str);
                end
                
                
                % find the widths and heights of peaks
                [obj.peakWidth(i,:), obj.peakHeight(i,:)] = obj.findPeakWH(sample);
                
            end
            
            % use linear regression, train the coefficients
            % model = 1 + 1/shift12 + 1/shift13 + 1/shift23
            if strcmp(obj.useModel, '1/t')
                model_ad = fitlm([obj.ratio./obj.time_shifts,...
                    1./obj.peakWidth, obj.peakHeight],...
                    obj.train_label(:,3));
            elseif strcmp(obj.useModel, 't')
                model_ad = fitlm(obj.time_shifts, obj.train_label(:,3));
            end
            % save a copy as property
            obj.model_ad = model_ad;
            
            obj.cleaned_train_label = train_label;
           
        end
        
        
        
        %===============================================================
        % test_raw_data is one single vehicle instance
        % assume LR model saved as a property
        % input: test_label [start_time, end_time]; num_test x 2
        % ouput est_speed; num_test x 1
        function est_speed = speedTest(obj,test_label)
            
            obj.test_label = test_label;
            
            est_speed = zeros(size(test_label,1),1);
            
            % for each test sample
            for i = 1:size(test_label,1)
                
                test_data = obj.extractMatrix(obj.raw_data,...
                    obj.test_label(i,1),obj.test_label(i,2));
            
                % find the time shifts
                if strcmp(obj.useShift, 'row')
                    shifts = obj.rowShifts( test_data );
                elseif strcmp(obj.useShift, 'time')
                    shifts = obj.timeShifts( test_data );
                else
                    error('invalid useShift argument');
                end
            
                % compute the estimated speed
                if strcmp(obj.useModel, '1/t')
                    est_speed(i,1) = predict(obj.model, obj.ratio./shifts);
                elseif strcmp(obj.useModel, 't')
                    est_speed(i,1) = predict(obj.model, shifts);
                end
            end
            
            obj.est_speed = est_speed;
            
        end
        
        
        
        %===============================================================
        % test_raw_data is one single vehicle instance
        % assume LR model saved as a property
        % input: test_label [start_time, end_time]; num_test x 2
        % ouput est_speed; num_test x 1
        function est_speed = speedTest_advanced(obj,test_label)
            
            obj.test_label = test_label;
            
            est_speed = zeros(size(test_label,1),1);
            
            % for each test sample
            for i = 1:size(test_label,1)
                
                test_data = obj.extractMatrix(obj.raw_data,...
                    obj.test_label(i,1),obj.test_label(i,2));
            
                % find the time shifts
                if strcmp(obj.useShift, 'row')
                    shifts = obj.rowShifts( test_data );
                elseif strcmp(obj.useShift, 'time')
                    shifts = obj.timeShifts( test_data );
                else
                    error('invalid useShift argument');
                end
                
                
                % find the width and peaks for the peaks
                [width, height] = obj.findPeakWH( test_data );
            
                % compute the estimated speed
                if strcmp(obj.useModel, '1/t')
                    
                    est_speed(i,1) = predict(obj.model_ad,...
                        [obj.ratio./shifts, 1./width, height ]);
                    
                elseif strcmp(obj.useModel, 't')
                    
                    est_speed(i,1) = predict(obj.model, shifts);
                    
                end
            end
            
            obj.est_speed = est_speed;
            
        end
        
        
        
        %===============================================================
        % evaluate training result
        % apply model on training data, and plot it versus true velocity
        function evaluateTrain(obj)
            
            trained_speed = zeros(size(obj.train_label,1),1);
            
            % for each training sample
            for i = 1:size(obj.train_label,1)
                
                % time shifts already saved
                shifts = obj.time_shifts(i,:);
                
                % compute the estimated speed
                trained_speed(i,1) = predict(obj.model, obj.ratio./shifts);
                
            end
            
            % plot result
            % also scatter each body
            x = 1:length(trained_speed);
            if ~iscolumn(x)
                x = x';
            end
            x_index = mod(x,4);
            raphi_index = (x_index == 1);
            yanning_index = (x_index == 2);
            emmanuel_index = (x_index == 3);
            brian_index = (x_index == 0);
            figure
            plot(obj.train_label(:,3),'b','LineWidth',2);
            hold on
            plot(trained_speed, 'r','LineWidth',2);
            
            % scatter points
            scatter(x(raphi_index), obj.train_label(raphi_index,3),'c*');
            scatter(x(raphi_index), trained_speed(raphi_index),'c*');
            
            scatter(x(yanning_index), obj.train_label(yanning_index,3),'b*');
            scatter(x(yanning_index), trained_speed(yanning_index),'b*');
            
            scatter(x(emmanuel_index), obj.train_label(emmanuel_index,3),'g*');
            scatter(x(emmanuel_index), trained_speed(emmanuel_index),'g*');
            
            scatter(x(brian_index), obj.train_label(brian_index,3),'k*');
            scatter(x(brian_index), trained_speed(brian_index),'k*');
            
            hold off
            legend('true speed', 'fitted speed');
            title('Evaluate training','FontSize',16);
            
            
            
        end
        
        
        
        
        %===============================================================
        % evaluate training result
        % apply model on training data, and plot it versus true velocity
        function plotTest(obj,true_speed)
            
            
            % plot result
            % also scatter each body
%             x = 1:length(obj.test_speed);
%             if ~iscolumn(x)
%                 x = x';
%             end
%             x_index = mod(x,4);
%             raphi_index = (x_index == 1);
%             yanning_index = (x_index == 2);
%             emmanuel_index = (x_index == 3);
%             brian_index = (x_index == 0);
            figure
            plot(true_speed,'b','LineWidth',2);
            hold on
            plot(obj.est_speed, 'r','LineWidth',2);
            
            % scatter points
%             scatter(x(raphi_index), obj.train_label(raphi_index,3),'c*');
%             scatter(x(raphi_index), trained_speed(raphi_index),'c*');
%             
%             scatter(x(yanning_index), obj.train_label(yanning_index,3),'b*');
%             scatter(x(yanning_index), trained_speed(yanning_index),'b*');
%             
%             scatter(x(emmanuel_index), obj.train_label(emmanuel_index,3),'g*');
%             scatter(x(emmanuel_index), trained_speed(emmanuel_index),'g*');
%             
%             scatter(x(brian_index), obj.train_label(brian_index,3),'k*');
%             scatter(x(brian_index), trained_speed(brian_index),'k*');
            
            hold off
            legend('true speed', 'estimated speed');
            xlabel('Samples','FontSize',18);
            ylabel('Speed (mph)','FontSize',18);
            title('Visualization of speed estimation','FontSize',16);
            
            
            
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
        
        
        
        
        %===============================================================
        % Use convolution to find the time shifts between three signal
        % vectors;
        % the sampling interval is not perfectly constant, and has outliers
        % Hence, use MATLAB time stamp
        % shifts: [shift12, shift13, shift23] in seconds
        % sample: num_sample x 4
        function shifts = timeShifts(obj, sample)
                
                % make sure that PIR1 peak appears last; then PIR2, PIR3
                PIR1 = sample(:,4);
                PIR2 = sample(:,3);
                PIR3 = sample(:,2);
                
                % use convolution to find the time shifts
                [~, ref_peak1] = max(PIR1); 
                ref_time1 = sample( ref_peak1, 1 );
                
                if strcmp(obj.useConv, 'zeroPadConv')
                    tmp_conv = obj.myconv_zero(PIR1, PIR2);
                elseif strcmp(obj.useConv, 'wrapConv')
                    tmp_conv = obj.myconv_wrap(PIR1, PIR2);
                else
                    error('invalid useConv argument');
                end
                [~, index_shift12] = max(tmp_conv);
                
                % when the data is not perfect, convolution will not give
                % the correct time shift
                if ref_peak1 + index_shift12 <= length(PIR1)
                    time2 = sample( ref_peak1 + index_shift12, 1);
                else
                    error('start_time: %0.12f, end_time: %0.12f',...
                                sample(1,1), sample(end,1));
                end
                
                if strcmp(obj.useConv, 'zeroPadConv')
                    tmp_conv = obj.myconv_zero(PIR1, PIR3);
                elseif strcmp(obj.useConv, 'wrapConv')
                    tmp_conv = obj.myconv_wrap(PIR1, PIR3);
                else
                    error('invalid useConv argument');
                end
                
                [~, index_shift13] = max(tmp_conv);
                
%                 if ref_peak1 + index_shift13 <= length(PIR1)
%                     time3 = sample( ref_peak1 + index_shift13, 1);
%                 else
%                     error('start_time: %0.12f, end_time: %0.12f',...
%                                 sample(1,1), sample(end,1));
%                 end
                
                % error handling
                try 
                    time3 = sample( ref_peak1 + index_shift13, 1);
                catch exception
                    
                    if ref_peak1 + index_shift13 > length(PIR1)
                        
                        str = sprintf('start_time: %0.12f, end_time: %0.12f',...
                                sample(1,1), sample(end,1));
                        disp(str);
                        
                    end
                    shifts = [];
                    return
                end

                
                % convert to seconds by multiplying 86400
                shifts(1,1) = (time2 - ref_time1)*86400;
                shifts(1,2) = (time3 - ref_time1)*86400;
                
                
                % not sure if the following is redundant
                if strcmp(obj.useConv, 'zeroPadConv')
                    tmp_conv = obj.myconv_zero(PIR2, PIR3);
                elseif strcmp(obj.useConv, 'wrapConv')
                    tmp_conv = obj.myconv_wrap(PIR2, PIR3);
                else
                    error('invalid useConv argument');
                end
                [~, index_shift23] = max(tmp_conv);
                
                if ref_peak1 + index_shift12 + index_shift23 <= length(PIR1)
                    time23 = sample( ref_peak1 + index_shift12 + index_shift23, 1);
                else
                    error('start_time: %0.12f, end_time: %0.12f',...
                                sample(1,1), sample(end,1));
                end
                
                shifts(1,3) = (time23 - time2)*86400;
                
                % save the MATLAB conv version
%                 tmp_conv = conv(PIR2, PIR2);
%                 [~, peak22] = max(tmp_conv);
%                 time22 = sample( floor((peak22+1)/2), 1);
%                 
%                 tmp_conv = conv(PIR2, PIR3);
%                 [~, peak23] = max(tmp_conv);
%                 time23 = sample( floor((peak22+1)/2) + (peak23-peak22), 1);
            
        end
        
        
        
        
        %===============================================================
        % Use convolution to find the time shifts between three signal
        % vectors;
        % the sampling interval is not perfectly constant, and has outliers
        % Hence, use MATLAB time stamp
        % shifts: [shift12, shift13, shift23] in rows
        % sample: num_sample x 4
        function shifts = rowShifts(obj, sample)
                
                % make sure that PIR1 peak appears last; then PIR2, PIR3
                PIR1 = sample(:,4);
                PIR2 = sample(:,3);
                PIR3 = sample(:,2);
                
                % use convolution to find the time shifts
                % shift between 1 2
                if strcmp(obj.useConv, 'zeroPadConv')
                    tmp_conv = obj.myconv_zero(PIR1, PIR2);
                elseif strcmp(obj.useConv, 'wrapConv')
                    tmp_conv = obj.myconv_wrap(PIR1, PIR2);
                else
                    error('invalid useConv argument');
                end
                
                if strcmp(obj.usePeak, 'max')
                    [~, index_shift12] = max(tmp_conv);
                elseif strcmp(obj.usePeak, 'halfThreshold')
                    index_shift12 = obj.myMassCenter(tmp_conv); 
                else
                    error('invalid usePeak argument');
                end
                
                
                % shift between 1 3
                if strcmp(obj.useConv, 'zeroPadConv')
                    tmp_conv = obj.myconv_zero(PIR1, PIR3);
                elseif strcmp(obj.useConv, 'wrapConv')
                    tmp_conv = obj.myconv_wrap(PIR1, PIR3);
                else
                    error('invalid useConv argument');
                end
                
                
                if strcmp(obj.usePeak, 'max')
                    [~, index_shift13] = max(tmp_conv);
                elseif strcmp(obj.usePeak, 'halfThreshold')
                    index_shift13 = obj.myMassCenter(tmp_conv); 
                else
                    error('invalid usePeak argument');
                end
                
                
                % shift between 2 3
                if strcmp(obj.useConv, 'zeroPadConv')
                    tmp_conv = obj.myconv_zero(PIR2, PIR3);
                elseif strcmp(obj.useConv, 'wrapConv')
                    tmp_conv = obj.myconv_wrap(PIR2, PIR3);
                else
                    error('invalid useConv argument');
                end
                
                
                if strcmp(obj.usePeak, 'max')
                    [~, index_shift23] = max(tmp_conv);
                elseif strcmp(obj.usePeak, 'halfThreshold')
                    index_shift23 = obj.myMassCenter(tmp_conv); 
                else
                    error('invalid usePeak argument');
                end
                
                % shift of rows
                shifts(1,1) = index_shift12;
                shifts(1,2) = index_shift13;
                shifts(1,3) = index_shift23;
                
        end
        
        
        
        %===============================================================
        % Find the width of peaks using resample or polynomial fitting
        % peakWidth, peakHeight: each 1x3
        % sample: num_sample x 4; first column is the time
        function [peakWidth, peakHeight] = findPeakWH(obj, sample)
                
                peakWidth = zeros(1,3);
                peakHeight = zeros(1,3);
                
                % Just to make sure that PIR1 peak appears last; then PIR2, PIR3
                PIR1 = sample(:,4);
                PIR2 = sample(:,3);
                PIR3 = sample(:,2);
                
                PIR = [PIR1, PIR2, PIR3]; % signals
                
                % call functions to compute the width and height of each
                % PIR signal
                for i=1:3
                    
                    [peakWidth(1,i), peakHeight(1,i)] = ...
                        computePeakWH_Fangyu( PIR(:,i), 12, 0.05, 10 );
                    
                end
                
                
                
        end
        
        
        
        %===============================================================
        % this function only resamples the signal and returns the width and
        % height, 
        
%         
%         function [peakW, peakH] = computePeakWH( signal, orgSampleRate, thre)
%             
%             % resample 
%             reSampledSignal;
%             
%             % apply 10% top
%             peakH;
%             
%             
%             % apply 10% low
%             
%             
%             % apply thres
%             width;
%             
%             %plot results and check
%             figure
%             plot(signal,'LineWidth',2);
%             hold on
%             scatter(1:length(signal),signal,'b*');
%             scatter(1:length(reSampledSignal), reSampledSignal, 'r.');
%             
%             
%         end
%         
%         
%         
        
        
        
        %===============================================================
        % matlab convolution function is confusing...
        % write own, instead of zero padding, circular shifts the data
        % input: signal vectors
        % ouput: time shifting row array. Max -> best aligned
        function conv_value = myconv_wrap(obj, signal1, signal2)
            
            if length(signal1) ~= length(signal2)
                error('Lengths of signals for convolution must be the same.\n');
            end
            
            m_signal2 = kron(ones(1,length(signal2)), signal2);
            % circular shift
            for i = 2:length(signal2)
                m_signal2(:,i) = circshift( m_signal2(:,i-1),-1 );
            end
            
            conv_value = signal1'*m_signal2;
        
        end
        
        
        
         %===============================================================
        % matlab convolution function is confusing...
        % write own, zero padding
        % ouput: time shifting row array. Max -> best aligned
        function conv_value = myconv_zero(obj, signal1, signal2)
            
            if length(signal1) ~= length(signal2)
                error('Lengths of signals for convolution must be the same.\n');
            end
            
            m_signal2 = kron(ones(1,length(signal2)), signal2);
            % circular shift
            for i = 2:length(signal2)
                m_signal2(:,i) = circshift( m_signal2(:,i-1),-1 );  % first shift
                m_signal2(end-i+2:end,i) = 0;   % zero padding
            end
            
            conv_value = signal1'*m_signal2;
        
        end
        
        
        
        %===============================================================
        % for debugging, plot signal data
        % input: start_time, end_time
        %       plot PIR1, PIR2, PIR3 data
        function plotSignalDuring(obj, t_start, t_end)
            
            plot_data = obj.extractMatrix(obj.raw_data,...
                t_start,t_end);
            
            figure
            plot(plot_data(:,2),'b','LineWidth',2);
            hold on
            plot(plot_data(:,3),'r','LineWidth',2);
            plot(plot_data(:,4),'g','LineWidth',2);
            hold off
            title(sprintf('%.8f to %.8f',t_start,t_end),'FontSize',20);
            xlabel('time','FontSize',16);
            ylabel('signal','FontSize',16);
            
        end
        
        
        
        %===============================================================
        % for debugging, plot signal data
        % shift and plot the detection window
        % input: initial start time, end time, shifts (s)
        % output: shifted start_time, end_time
        function [shifted_start_time, shifted_end_time] = shiftSignalFrom(obj,...
                t_start, t_end, start_shift, end_shift)
            
            % find the exact time from the data
            ini_plot_data = obj.extractMatrix(obj.raw_data,...
                t_start,t_end);
            shifted_plot_data = obj.extractMatrix(obj.raw_data,...
                t_start + start_shift/86400,t_end + end_shift/86400);
            
            plot_data = obj.extractMatrix(obj.raw_data,...
                min(t_start, t_start + start_shift/86400),...
                max(t_end, t_end + end_shift/86400));
            
            % original time interval
            ini_start_index = find(plot_data(:,1) == ini_plot_data(1,1));
            ini_end_index = find(plot_data(:,1) == ini_plot_data(end,1));
            
            shifted_start_index = find(plot_data(:,1) == shifted_plot_data(1,1));
            shifted_end_index = find(plot_data(:,1) == shifted_plot_data(end,1));
            
            figure
            plot(plot_data(:,2),'b','LineWidth',2);
            hold on
            plot(plot_data(:,3),'r','LineWidth',2);
            plot(plot_data(:,4),'g','LineWidth',2);
            yL = get(gca,'YLim');
            plot([ini_start_index, ini_start_index],[yL(1), yL(2)],...
                'k','LineWidth',2,...
                'DisplayName','initial start');
            plot([ini_end_index, ini_end_index],[yL(1), yL(2)],...
                'k','LineWidth',2,...
                'DisplayName','initial end');
            plot([shifted_start_index, shifted_start_index],[yL(1), yL(2)],...
                'b--','LineWidth',2,...
                'DisplayName','shifted start');
            plot([shifted_end_index, shifted_end_index],[yL(1), yL(2)],...
                'b--','LineWidth',2,...
                'DisplayName','shifted end')
            hold off
            title(sprintf('%.8f to %.8f',t_start,t_end),'FontSize',20);
            xlabel('time','FontSize',16);
            ylabel('signal','FontSize',16);
            
            shifted_start_time = shifted_plot_data(1,1);
            shifted_end_time = shifted_plot_data(end,1);
            
            
        end
        
        
        
        %===============================================================
        % for debugging, plot particular detections in training data
        % input: detections
        %       plot PIR1, PIR2, PIR3 data
        % output: start time and end time of each detection, in MATLAB
        function [start_time, end_time] = plotSignalInDetection(obj, det)
            
            for i = 1:length(det)
            
            t_start = obj.train_label(det(i),1);
            t_end = obj.train_label(det(i),2);
            
            % save the times
            start_time(i,1) = t_start;
            end_time(i,1) = t_end;
            
            plot_data = obj.extractMatrix(obj.raw_data,...
                t_start,t_end);
            
            figure
            plot(plot_data(:,2),'b','LineWidth',2);
            hold on
            plot(plot_data(:,3),'r','LineWidth',2);
            plot(plot_data(:,4),'g','LineWidth',2);
            hold off
            title(sprintf('Detection %d \n shifts(s): 12:%.4f, 13:%.4f, 23:%.4f',...
                det(i), obj.time_shifts(det(i),1),...
                obj.time_shifts(det(i),2),...
                obj.time_shifts(det(i),3)),'FontSize',20);
            xlabel('time','FontSize',16);
            ylabel('signal','FontSize',16);
            
            % output time shifts
%             str = sprintf('shift12: %.4f \n shift13: %.4f \n shift23 \n',...
%                            obj.time_shift(det(i),1),...
%                            obj.time_shift(det(i),2),...
%                            obj.time_shift(det(i),3));
%             
%             h = text(floor(length(noisyFlow)/3),...
%                     1*max(noisyFlow),...
%                     str );
%                 set(h, 'FontSize',16);
        
                
            end
        end
        
        
        
        %===============================================================
        % for debugging, clearn the not well marked training labels
        % shift the 
        % input: old and new detection time intervals
        %       save the training label in the cleaned_label property
        function replaceDetectionLabel(obj,...
                old_interval, new_interval)
            
            % find the exact time from the data
            if sum(obj.cleaned_train_label(:,1) == old_interval(1)) ~=0
                
                index = ( obj.cleaned_train_label(:,1) == old_interval(1) );
                obj.cleaned_train_label(index, 1) = new_interval(1);
                
                index = ( obj.cleaned_train_label(:,2) == old_interval(2) );
                obj.cleaned_train_label(index, 2) = new_interval(2);
                
            else
                disp('The train data does not have the specified time\n');
            end
            
        end
        
        
        
        %===============================================================
        % center of mass of signal
        function center_mass = myMassCenter(obj,vec)
            
            b = 1:length(vec);
            
            if ~iscolumn(vec)
                vec = vec';
            end
            
            % remove negative values since we know the shifts should be
            % positive
            % if we further only care about the peak
            vec(vec< max(vec)/2) = 0;
            
            center_mass = b*vec/sum(vec);
            
        end
        
        
    end
    
end













