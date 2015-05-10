classdef cross_validation < handle
	properties
		cv_times
		raw_data
		labels
	end

	methods
		%dummy constructor
		function obj = cross_validation()
			disp('Loading data...')
			%The times that mark the boundary between folds
			obj.cv_times = [735918.436,735918.4425,735918.4494, 735918.456, 735918.4632, 735918.47, 735918.477, 735918.4832, 735918.4894, 735918.4963, 735918.5039];


			%load raw data
			obj.raw_data = dlmread('../Data/filteredAllData_num.csv');

			%load ground truth labels
			obj.labels = dlmread('../Data/cleaned_labels_num_v2_4.csv');
		end

		%Produces a training set and test set for a particular fold (1-9)
		%data - could be the raw data or the labels
		%num - the fold ID.  In other words, which of the folds will be treated as the test set?
		function [train,test] = get_fold(obj, data, num, time_col_id)
            %[obj.cv_times(num), obj.cv_times(num+1)]
			%Select the range of data that is in the training and test set
			test_selection = data(:,time_col_id) >= obj.cv_times(num) & data(:,time_col_id) <= obj.cv_times(num+1);
			train_selection = (data(:,time_col_id) < obj.cv_times(num) | data(:,time_col_id) > obj.cv_times(num+1)) & data(:,time_col_id) < obj.cv_times(10);
	
			%assign this portion of the data into the return values
			test = data(test_selection,:);
			train = data(train_selection,:);

        end
        
        
        
        %===========================================================
        % Vehicle detection
        
        % This function exhausts and compares different vehicle detection
        % models.
        % If only test a specific model, set the parameters and use
        % run_detection_cv instead.
        function output_tab = run_several_detection_models(obj)
            %modelTypes = {'gmm', 'logit'};
            output_tab = zeros(50,6);
            %distcomp.feature('LocalUseMpiexec', false)
            % parpool('open', 4);
            disp('running CV')
            for w = 1:50
                    w
                    s = ceil(w/5);
                    tic
                    [total_false_pos_gmm, total_false_neg_gmm] = obj.run_detection_cv(w, s, 'gmm');
                    toc
                    
                    tic
                    [total_false_pos_gmmp, total_false_neg_gmmp] = obj.run_detection_cv(w, s, 'gmmp');
                    toc
                    
                    tic
                    [total_false_pos_logit, total_false_neg_logit] = obj.run_detection_cv(w, s, 'logit');
                    toc
                    
                    output_row = [total_false_pos_gmm, total_false_neg_gmm,...
                        total_false_pos_gmmp, total_false_neg_gmmp,...
                        total_false_pos_logit, total_false_neg_logit];
                    output_tab(w,:) = output_row;
            end
            % parpool('close');

            % produce plot of window size vs. error, for several models
            
            f = figure;
            set(f,'Units','normalized');
            set(f,'Position',[0 0 1 0.5]);
            plot(1:50, output_tab(:,1), 'b--','LineWidth',2);
            hold on
            plot(1:50, output_tab(:,2), 'b-','LineWidth',2);
            plot(1:50, output_tab(:,3), 'g--','LineWidth',2);
            plot(1:50, output_tab(:,4), 'g-','LineWidth',2);
            plot(1:50, output_tab(:,5), 'r--','LineWidth',2);
            plot(1:50, output_tab(:,6), 'r-','LineWidth',2);
            leg = legend('GMM-FP','GMM-FN', 'GMMP-FP', 'GMMP-FN', 'Logit-FP','Logit-FN');
            set(leg,'FontSize',18);
            title('Comparison of Detection Models','FontSize',18);
            xlabel('Window Size','FontSize',18);
            ylabel('Number of Errors','FontSize',18);
            hold off
            
        end
        
        
        % This function is used for vehicle detection.
        function [total_false_pos, total_false_neg] = run_detection_cv(obj, window, shift, modeltype)
            
            use_prior = 'no';
            if(strcmp(modeltype,'gmmp'))
                use_prior='yes';
                modeltype='gmm';
            end
            
            detector = vehDet_v3(window, shift, modeltype, use_prior);
            total_false_pos = 0;
            total_false_neg = 0;
            
            % For test purpose, just use one
            for fold_id = 1:1
                %Split the raw data and labels into training set, test set
                [train, test] = obj.get_fold(obj.raw_data, fold_id, 1);
                [trainlabels, testlabels] = obj.get_fold(obj.labels, fold_id, 2);
               
                detector.detectionTrain(train, trainlabels(:,2:3));
                
                %use the model to make predictions
                predictions = detector.vehTest(test);
                %disp('Predictions')
                %predictions(1,:)
                %predictions(end,:)
                %testlabels(:,2:3)
                %predictions
                [false_pos, false_neg] = obj.count_detection_mistakes(testlabels(:,2:3), predictions);
                total_false_pos = total_false_pos + false_pos;
                total_false_neg = total_false_neg + false_neg;
                
                %obj.plot_detection_results(test, testlabels, predictions);
                
            end
        end
        
        
        % This function returns the FP and FN
		function [false_pos, false_neg] = count_detection_mistakes(obj, ground_truth, predictions)

			false_pos = 0;
			%Find false positives - entries in predictions that have no match in the ground truth
			for i = 1:size(predictions,1)
				%Find the matches
				matches = ground_truth(:,2) > predictions(i,1) & ground_truth(:,1) < predictions(i,2);
				num_matches = sum(matches);
				if num_matches ~= 1
					false_pos = false_pos + 1;
				end
			end

			false_neg = 0;
			%Find false negatives - entries in the ground truth that have no match in the predictions
			for i = 1:size(ground_truth, 1)
				matches = predictions(:,2) > ground_truth(i,1) & predictions(:,1) < ground_truth(i,2);
				num_matches = sum(matches);
				if num_matches ~= 1
					false_neg = false_neg + 1;
				end
			end

        end

        
        
        % This function plots the detection results
        function plot_detection_results(obj, test_data, test_labels, predictions)
            
            t = test_data(:,1);
            
            figure
            plot(t, test_data(:,2),'b','LineWidth',2);
            hold on
            plot(t, test_data(:,3),'r','LineWidth',2);
            plot(t, test_data(:,4),'g','LineWidth',2);
            
            title('Detection visualization');
            xlabel('time','FontSize',16);
            ylabel('signal','FontSize',16);
            
            % plot true labels
            yl = ylim;
            for i=1:size(test_labels,1)
                r = rand(1);
                plot([test_labels(i,2), test_labels(i,3)], [yl(2)/3 + r, yl(2)/3 + r],'k',...
                    'LineWidth',2);
            end
            
            for i = 1: size(predictions,1)
                r = rand(1);
                plot([predictions(i,1), predictions(i,2)], [2*yl(2)/3+r, 2*yl(2)/3+r],'r',...
                    'LineWidth',2);
                
            end
            
            hold off
            
        end
        
        
        %========================================================
        % The following code is for speed estimation
        
        % we used several different speed estimation models.
        % This function exhausts all different models and compute the RMS
        % for each model
        % Only useful when comparing different models, otherwise set
        % parameters and call run_speed_cv directly
        function run_several_speed_models(obj)
            
            useConv = {'zeroPadConv', 'wrapConv'};
            usePeak = {'max', 'halfThreshold'};
            modelType = {'t', '1/t'};
            
            for i = 1:2
                for j = 1:2
                    for k = 1:2
                        [useConv{i}, ' -- ', usePeak{j}, ' -- ',  modelType{k}];
                        rms = obj.run_speed_cv(useConv{i}, 'row', usePeak{j}, modelType{k});
                    end
                end
            end
            
        end
		
		%Actually runs the cross-validation and computes the error
		%returns the root mean squared (rms) error
		function [rms, pred_speed, true_speed] =...
                run_speed_cv(obj, useConv, useShift, usePeak, modelType)
            
			speed_est = speedEst(obj.raw_data);

			sum_of_squares = 0;
			prediction_count = 0;

			%run through all of the folds and save the predicted speed
            % for plotting
            pred_speed = [];
            true_speed = [];
            
            % For test reason, just use one now.
            for fold_id = 1:9
                %split the labels into training and test set
                [train, test] = obj.get_fold(obj.labels, fold_id, 2);
                
                % train the model
                fold_id
                %speed_est.speedTrain(train(:,2:4), useConv,...
                %    useShift, usePeak, modelType);
                speed_est.speedTrain_advanced(train(:,2:4));
                
                %use the model to make predictions on the test data
                %predictions = speed_est.speedTest(test(:,2:3));
                predictions = speed_est.speedTest_advanced(test(:,2:3));
                
                pred_speed = [pred_speed; predictions];
                
                % plot the result
                % speed_est.plotTest(test(:,4));
                
                true_speed = [true_speed; test(:,4)]; 
                
                % determine the error of this fold and add it to the sum_of_squares
                this_fold_error = sum((predictions - test(:,4)).^2);
                sum_of_squares = sum_of_squares + this_fold_error;
                
                % also increase the count
                prediction_count = prediction_count + size(test,1);
            end
            
            %compute RMS error
            rms = sqrt(sum_of_squares / prediction_count);
            
            
			
        end
        

        
       
	end
end


