% class unsup_vehDet
% Yanning, 
% unsupervised approach for vehicle detection


classdef unsup_vehDet < handle
    
    properties
        
        % raw data 
        raw_data;
        
        % filtered data, removed ambient temperature
        filtered_data;
        
        % averaged data
        avg_filtered_data;
        avg_raw_data;
        
        % window size and step size
        T;
        s;
        
        % sampling rate
        sampling_rate;
        
        % gaussian mixture model
        gm;
        idx;
        cluster_veh;
        cluster_nonveh;
        cluster_boundary;
        
        % MAP
        u_veh;
        sig_veh;
        
        u_nonveh;
        sig_nonveh;
        
        % Assume the following three parameters are known
        % they are computed from the entire filtered data 
        % before applying any model
        cov_likelihood_veh;
        cov_likelihood_nonveh;
        nonveh_perc;   
        
        % detected interval
        det_interval;
        
        
    end
    
    
    methods
        %===============================================================
        % initialize object properties with all the raw data
        function obj = unsup_vehDet()
            
            %load raw data
			obj.raw_data = dlmread('./unfilteredAllData_num.csv');
            obj.filtered_data = dlmread('./filteredAllData_num.csv');
            
            % take only the first a few cycles 73000
            obj.raw_data(73001:end,:) = [];
            obj.filtered_data(73001:end,:) = [];
            
            obj.raw_data(1:10000,:) = [];
            obj.filtered_data(1:10000,:) = [];
            
            % default
            obj.sampling_rate = 12;
            
            % set empty
            obj.cluster_veh = [];
            obj.cluster_nonveh = [];
            obj.cluster_boundary = [];
            
            obj.det_interval = zeros(0,2);
            
        end
        
        %===============================================================
        % plot the original data
        function obj = plot_temp_distr(obj, pir_index)
            
            % set up the time vector
            time = 1:size(obj.raw_data,1);
            time = time/(obj.sampling_rate*60); % in min
            
            for i = 1:length(pir_index)
                
                PIR1_raw = obj.raw_data(:,pir_index(i)+1);
                PIR1_filtered = obj.filtered_data(:,pir_index(i)+1);
                
                figure
                plot(time, PIR1_raw, 'r')
                hold on
                grid on
                plot(time, PIR1_filtered, 'b')
                legend(sprintf('pir %d raw data',pir_index(i)),...
                    sprintf('pir %d filtered data',pir_index(i)) )
                title('time series data','FontSize', 20)
                xlabel('time (min)', 'FontSize', 18)
                ylabel('temperature C', 'FontSize', 18)
            end
            
            
        end
        
        
        %===============================================================
        % GMM fitting
        % 1. operate on the filtered data
        % 2. assume the ambient and the car both follows a gaussian 
        % 3. use a mixture gaussian for clustering
        % 4. try on both the point or windowed
        % input: signal that to be fitted n samples x k dimensions
        %        num_clusters: number of clusters
        % output: an object of gm
        function gmm_fit(obj, signal, num_clusters)
            
            options = statset('Display','final');
            
            if size(signal,2) < 4
                sprintf('Warning, signal should include time colume')
            end
            
            obj.gm = fitgmdist(signal(:,[2,3,4]), num_clusters,...
                'Options', options);
            
        end
        
        
        %===============================================================
        % plot fitting result
        function gmm_cluster(obj, signal, plot_option)
            
            if size(signal,2) < 4
                
                sprintf('Warning, signal should include time column')
                
            end
            
            obj.idx = cluster(obj.gm,signal(:,[2,3,4]));
            
            % the small cluster is the veh_cluster
            if sum(obj.idx==1) <= sum(obj.idx==2)
                obj.cluster_veh = signal(obj.idx == 1,:);
                obj.cluster_nonveh = signal(obj.idx == 2,:);
                
                % switch idx to make it consistent
                % 1-nonveh; 2-veh; 0-boundary
                obj.idx(obj.idx == 1) = 3;
                obj.idx(obj.idx == 2) = 1;
                obj.idx(obj.idx == 3) = 2;
            else
                obj.cluster_veh = signal(obj.idx == 2,:);
                obj.cluster_nonveh = signal(obj.idx == 1,:);
            end

            if strcmp(plot_option, 'plot')
                
                % only for debugging
                if size(signal,2) == 3
                    figure
                    scatter(signal(:,2), signal(:,3),'.')
                    hold on
                    ezcontour(@(x,y)pdf(obj.gm,[x y]));
                end
                
                if size(signal,2) == 3
                    % only for debugging
                    figure
                    h1 = scatter(obj.cluster_veh(:,2),...
                                 obj.cluster_veh(:,3),10,'r.');
                    hold on
                    h2 = scatter(obj.cluster_nonveh(:,2),...
                                 obj.cluster_nonveh(:,3),10,'k.');
                    legend([h1 h2],'Cluster veh','Cluster nonveh','Location','NW');
                    title('fitting of signal','FontSize',20)
                    
                elseif size(signal,2) == 4
                    
                    figure
                    h1 = scatter3(obj.cluster_veh(:,2),...
                                  obj.cluster_veh(:,3),...
                                  obj.cluster_veh(:,4),10,'r.');
                    hold on
                    h2 = scatter3(obj.cluster_nonveh(:,2),...
                                  obj.cluster_nonveh(:,3),...
                                  obj.cluster_nonveh(:,4),10,'k.');
                    legend([h1 h2],'Veh','Nonveh','Location','NW');
                    title('fitting of signal','FontSize',20)
                    
                end
                
                
                figure
                hold on
                len = size(signal,1);
                x_index = 1:len;
                if sum(obj.idx==1) <= sum(obj.idx==2)
                    x_index_veh = x_index(obj.idx == 1);
                    x_index_nonveh = x_index(obj.idx == 2);
                else
                    x_index_veh = x_index(obj.idx == 2);
                    x_index_nonveh = x_index(obj.idx == 1);
                end
                
                for i = 2 : size(signal,2)
                    
                    scatter(x_index_veh, obj.cluster_veh(:,i), 'r.');
                    scatter(x_index_nonveh, obj.cluster_nonveh(:,i), 'k.');
                    
                end
                
                plot(x_index, signal(:,2), 'g');
                plot(x_index, signal(:,3), 'b');
                plot(x_index, signal(:,4), 'm');
                
            end
            
        end
        
        
        %===============================================================
        % model 1
        % stream data in, upon arrival of every new data point,
        % re-fit all data with GMM
        % input: new data packet size pkt_size
        %        data_type: avg or point
        % output: refreshing figures
        function gmm_model_1(obj, pkt_size, data_type)
                       
            end_data = 0;
            
            % choose the stream data type
            if strcmp( data_type, 'point')
                stream_data = obj.raw_data;
            elseif strcmp( data_type, 'avg')
                stream_data = obj.avg_raw_data;
            else
                error('invalid pre_option')
            end
            
            % figure
            scrsz = get(groot, 'ScreenSize' );
            h = figure('Position', [1 scrsz(4)/2 scrsz(3) scrsz(4)/2]);
            title('Online Detection','FontSize',20)
            % stream data
            end_data = end_data + pkt_size;
            while end_data <= size(stream_data,1)
                
                disp(end_data)
                % fit data
                obj.gmm_fit( stream_data(1:end_data,:), 2);
                
                % cluster data
                obj.gmm_cluster( stream_data(1:end_data,:), 'no_plot')
                
                % refresh the figure
                obj.refresh_fig(h)
                
                % pause(0.1)
                
                % move on
                end_data = end_data + pkt_size;
                
            end
            
            
        end
        
        
        %===============================================================
        % refresh figure
        % plot the time series clustering of the result
        % input: fig_h, the figure handle
        function refresh_fig(obj, fig_h)
            
            % plot on the same figure
            figure(fig_h)
            
            len = size(obj.cluster_veh,1) +...
                size(obj.cluster_nonveh,1) + ...
                size(obj.cluster_boundary,1);
            
            x_index = 1:len;
            % 1-nonveh; 2-veh; 0-boundary
            x_index_nonveh = x_index(obj.idx == 1);
            x_index_veh = x_index(obj.idx == 2);
            x_index_boundary = x_index(obj.idx == 0);
            
            for i = 2 : size(obj.cluster_veh,2)
                if ~isempty(x_index_veh)
                    h_veh = scatter(x_index_veh, obj.cluster_veh(:,i), 'r.');
                end
                hold on
                if ~isempty(x_index_nonveh)
                    h_nonveh = scatter(x_index_nonveh,...
                        obj.cluster_nonveh(:,i), 'k.');
                end
                if ~isempty(x_index_boundary)
                    h_boundary = scatter(x_index_boundary, ...
                        obj.cluster_boundary(:,i), 'b.');
                end
            end
            
            % set x axis as 100 s: 1200 sample
            if ~isempty(x_index_boundary)
                legend([h_veh, h_nonveh, h_boundary],...
                    'Veh','Nonveh','Boundary','Location','NW');
            else
                legend([h_veh, h_nonveh],...
                    'Veh','Nonveh','Location','NW');
            end
            
            xaxis_end = len;
            % xaxis_start = xaxis_end - 3600;
            xaxis_start = 1;
            xlim([xaxis_start, xaxis_end]);
            
            hold off
            
        end
        
            
        %===============================================================
        % model 2
        % - Use bayesian model to estimate the posterior distribution of
        % the mean of the two clusters. 
        % -Then upon the incoming of new data, use the posterior 
        % distribution of u for two clusters to cluster the new data.
        % - Both u are modeled as gaussian
        % - Likely hood modeled as gaussian
        % - The variance of two clusters are assumed to be known (which
        % actually depends on the sensor error, and variety of cars)
        % the prior is obtained from GMM
        % input: calib_time: calibration time in min
        %        pkt_interval: time for updating model in min
        %        data_type: 'point' or 'avg'; original or windowed
        %        cut_prob: if y is cut_prob (e.g. 60%) likely to be veh,
        %                 then put in veh; if 50%-50% mark as boundary pts
        %        update: 'update', 'no_update' whether update parameters
        function MAP_model(obj, calib_time, pkt_interval, data_type, ...
                           cut_prob, update)
            
            %=========================================
            % find the covariance matrix of two cluster 
            % This can be obtained from the filtered data
            [obj.cov_likelihood_nonveh,...
                obj.cov_likelihood_veh, obj.nonveh_perc] = obj.get_cov();
            
            %=========================================
            % calibrate prior from first a few min data
            obj.get_prior(calib_time, data_type);
            

            %=========================================
            % refresh figure
            % figure
            scrsz = get(groot, 'ScreenSize' );
            h = figure('Position', [1 scrsz(4)/2 scrsz(3) scrsz(4)/2]);
            title('Online Detection','FontSize',20)
            obj.refresh_fig(h)
            
            
            %=========================================
            % choose the stream data type
            if strcmp( data_type, 'point')
                stream_data = obj.raw_data;
                step_rate = obj.sampling_rate;  % points per second
            elseif strcmp( data_type, 'avg')
                stream_data = obj.avg_raw_data;
                step_rate = obj.sampling_rate/obj.s;    % may not be int
            else
                error('invalid data_type option')
            end
            
            % Assume the sensor calibrate for calib_time min = 5*60*12 =
            % 1800
            start_data = floor(calib_time*60*step_rate) + 1;
            pkt_size = floor(pkt_interval*60*step_rate);
            
            
            end_data = start_data + pkt_size;
            while end_data <= size(stream_data,1)
                
                %=========================================
                % cluster the new pkt points to veh or nonveh
                pkt_data = stream_data(start_data : end_data, :);
                
                f_nonveh = mvnpdf( pkt_data(:, [2,3,4]),...
                            obj.u_nonveh, obj.sig_nonveh);
                f_veh = mvnpdf( pkt_data(:, [2,3,4]),...
                            obj.u_veh, obj.sig_veh);
                
                % multiply by cluster prior and then normalize
                f_nonveh = obj.nonveh_perc*f_nonveh;
                f_veh = (1-obj.nonveh_perc)*f_veh;
                
                p_nonveh = f_nonveh./(f_nonveh+f_veh);
                p_veh = f_veh./(f_nonveh+f_veh);
                
                % 1-nonveh; 2-veh; 0-boundary pts
                pkt_idx = zeros(length(p_nonveh),1);
                pkt_idx(p_nonveh >= cut_prob) = 1;
                pkt_idx(p_veh >= cut_prob) = 2;
                
                % update cluster information in properties
                obj.idx = [obj.idx; pkt_idx];
                obj.cluster_nonveh = [obj.cluster_nonveh;...
                                      pkt_data(pkt_idx==1,:)];
                obj.cluster_veh = [obj.cluster_veh;...
                                      pkt_data(pkt_idx==2,:)];
                obj.cluster_boundary = [obj.cluster_boundary;...
                                      pkt_data(pkt_idx==0,:)];                  
                
                %=========================================
                % refresh figure
                obj.refresh_fig(h)
                                
                %=========================================
                % compute the posterior distribution of two clusters for
                % clustering the next pkt data
                if strcmp(update, 'update')
                    disp([obj.u_nonveh; obj.u_veh])
                    
                    [u_nonveh_post, sig_nonveh_post] =...
                        obj.update_posterior(obj.u_nonveh, obj.sig_nonveh,...
                        obj.cov_likelihood_nonveh,...
                        pkt_data(pkt_idx==1,:));
                    obj.u_nonveh = u_nonveh_post';
                    obj.sig_nonveh = sig_nonveh_post;
                    
                    % try to not update vehicle cluster
                    [u_veh_post, sig_veh_post] =...
                        obj.update_posterior(obj.u_veh, obj.sig_veh,...
                        obj.cov_likelihood_veh,...
                        pkt_data(pkt_idx==2,:));
                    obj.u_veh = u_veh_post';
                    obj.sig_veh = sig_veh_post;
                    
                    disp([u_nonveh_post'; u_veh_post'])
                    
                end
                
                % update stream
                start_data = end_data+1;
                end_data = end_data + pkt_size;
                
                
            end
            
            
            %=========================================
            % mark detection of cars
            % now assume minimal presence of a car is at least 1s
            obj.find_det_interval( step_rate/2, h)
            
        end
        

        %===============================================================
        % In model MAP, we need covariance matrix, which partially reflects
        % our confidence to the sensor and knowledge to the traffic
        % output: covariance matrix and percentage of nonveh points
        function [cov_nonveh, cov_veh, nonveh_perc] = get_cov(obj)
            
            options = statset('Display','final');
            
            gmm = fitgmdist(obj.filtered_data(:,[2,3,4]), 2,...
                'Options', options);
            mix_prop = gmm.ComponentProportion;
            
            nonveh_perc = max(mix_prop);
            
            index = cluster(gmm, obj.filtered_data(:,[2,3,4]) );
            
            if sum(index == 1) >= sum(index == 2)
                % first cluster is nonveh
                % find the covariance
                nonveh = obj.filtered_data(index==1,:);
                veh = obj.filtered_data(index==2,:);
            else 
                nonveh = obj.filtered_data(index==2,:);
                veh = obj.filtered_data(index==1,:);
            end
            
            cov_nonveh = cov(nonveh(:,[2,3,4]));
            cov_veh = cov(veh(:,[2,3,4]));
            
            
        end
        
       
        %===============================================================
        % For model MAP
        % universal function for updating the posterior distribution of
        % Gaussian likelihood and prior
        % input: prior of u; likelihood sigma; and data: n samples x dim
        function [u_post, sig_pred] = update_posterior(obj, u_prior, sig_prior,... 
                                                        sig_likelihood,...
                                                        data)
            
            % u_prior should be 1x3
            % but the computation takes column vector
            if ~iscolumn(u_prior)
                u_prior = u_prior';
            end
                                               
            % compute the data mean
            u_data = mean(data(:,[2,3,4]),1);
            u_data = u_data';
            len_data = size(data,1);
            
            % compute posterior mean
            inv_sig_prior = inv(sig_prior);
            inv_sig_likelihood = inv(sig_likelihood);
            u_post = (inv_sig_prior + len_data*inv_sig_likelihood)...
                     \(sig_prior\u_prior + sig_likelihood\u_data*len_data);
            
            sig_post = inv(inv_sig_prior + len_data*inv_sig_likelihood);
            
            sig_pred = sig_post + sig_likelihood;
            
            
        end
        
        %===============================================================
        % For model MAP, calibrate prior distribution
        function get_prior(obj, calib_time, data_type)
            
            % choose the stream data type
            if strcmp( data_type, 'point')
                stream_data = obj.raw_data;
                step_rate = obj.sampling_rate;  % points per second
            elseif strcmp( data_type, 'avg')
                stream_data = obj.avg_raw_data;
                step_rate = obj.sampling_rate/obj.s;    % may not be int
            else
                error('invalid data_type option')
            end
            
            % Assume the sensor calibrate for calib_time min = 10*60*6 = 3600
            end_data = floor(calib_time*60*step_rate);
            
            % fit gmm model to get the prior parameters
            % make sure this is a good clustering, otherwise wait for more
            % data
            prior_data = stream_data(1:end_data,:);
            obj.gmm_fit( prior_data, 2);
            mix_prop = obj.gm.ComponentProportion;
            
            if mix_prop(1) > mix_prop(2)
                % first cluster is non_veh
                obj.u_nonveh = obj.gm.mu(1,:);
                obj.u_veh = obj.gm.mu(2,:);
                
                obj.sig_nonveh = obj.gm.Sigma(:,:,1);
                obj.sig_veh = obj.gm.Sigma(:,:,2);
            else
                % second cluster is nonveh
                obj.u_nonveh = obj.gm.mu(2,:);
                obj.u_veh = obj.gm.mu(1,:);
                
                obj.sig_nonveh = obj.gm.Sigma(:,:,2);
                obj.sig_veh = obj.gm.Sigma(:,:,1);
            end
            

            % save clustering result.
            % Those result may be bad
            index = cluster(obj.gm, prior_data(:,[2,3,4]) );
            
            if sum(index == 1) >= sum(index == 2)
                % first cluster is nonveh
                % find the covariance
                obj.cluster_nonveh = prior_data(index==1,:);
                obj.cluster_veh = prior_data(index==2,:);
                obj.idx = index;
            else 
                obj.cluster_nonveh = prior_data(index==2,:);
                obj.cluster_veh = prior_data(index==1,:);
                
                % switch 1 and 2
                index(index==1) = 3;
                index(index==2) = 1;
                index(index==3) = 2;
                obj.idx = index;
            end
            
        end
        
        
        %===============================================================
        % find detection time
        % output the time intervel of the detected presence of cars
        % input: the min_interval of car in points; 
        %        e.g. the interval must be longer
        %             than min_interval to be considered as a car
        %        h_fig: the figure handle
        function find_det_interval(obj, min_interval, h_fig)
            
            obj.det_interval = zeros(0,2); % start_row, end_row
            
            % construct a matrix 
            % [start_index, end_index, data]
            tmp_index = obj.idx;
            % 1-nonveh; 2-veh; 0-neither
            tmp_index(tmp_index ~= 2) = 0;
            
            
            tic
            i_start = 1;
            i_end = 1;
            det_flag = 0;
            
            while i_end <= length(tmp_index)
                
                % if not detected, keep moving end index
                if tmp_index(i_end) == 0 && det_flag == 0
                    i_end = i_end +1;
                    continue
                end
                
                % if new detection, mark start
                if tmp_index(i_end) == 2 && det_flag == 0
                    i_start = i_end;
                    det_flag = 1;
                    i_end = i_end+1;
                    continue
                end
                
                % if detected, trying to find end of detection
                if tmp_index(i_end) == 0 && det_flag == 1
                    
                    obj.det_interval = [obj.det_interval; [i_start i_end-1]];
                    det_flag = 0;
                    i_end = i_end+1;
                    continue
                end
                
                % if in detection
                if tmp_index(i_end) == 2 && det_flag == 1
                    i_end = i_end+1;
                    continue
                end

            end
            
            
            short_det_index = (obj.det_interval(:,2) - obj.det_interval(:,1)...
                                < min_interval);
                            
            % remove the false detection
            obj.det_interval(short_det_index,:) = [];
            
            toc
                                
            % plot detection results in the figure
            figure(h_fig)
            for i = 1:size(obj.det_interval,1)
                hold on
                plot([obj.det_interval(i,1), obj.det_interval(i,2)], ...
                     [5, 5], 'Color', [0,0.5,0], 'LineWidth',2)
            end
            
            hold off
            
        end
        
        %===============================================================
        % generate the avg data
        % input: T, window size; s, step size
        % output: average data saved in properties
        function avg_data(obj, T, s)
            
            obj.T = T;
            obj.s = s;
            
            for i = 1: size(obj.raw_data,2)
                
                % also apply hamming window on the time stamps
                % should give the time of the window
                obj.avg_raw_data(:,i) = obj.average_signal(obj.raw_data(:,i));
                obj.avg_filtered_data(:,i) = obj.average_signal(...
                                                obj.filtered_data(:,i));
                
            end
            
        end
       

        %===============================================================
        % average a signal using a Hamming window
        % input: signal, window size T, step size s
        % output: averaged value of signal
        function avgVal = average_signal(obj, signal)
            
            % second approach
            numWindows = floor( (size(signal, 1)-obj.T)/obj.s)+1;
            
            chopped_signal = zeros(numWindows, obj.T);
            
            for i = 1:numWindows
                
                chopped_signal(i,:) = signal( (i-1)*obj.s+1 : (i-1)*obj.s + obj.T )';
                
            end
            
            % apply window
            avgVal = chopped_signal*hamming(obj.T)/sum(hamming(obj.T));


            
        end
        
        
    end
    
end

















