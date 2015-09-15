% function for using threshold to detect vehicles

function vehDet_threshold(dataOut)

% filtered PIR data
% only use the test peiord
dataOut_test = dataOut;
dataOut_test(1:10000,:) = [];
PIR1_filtered = dataOut_test(:,1);
PIR2_filtered = dataOut_test(:,2);
PIR3_filtered = dataOut_test(:,3);

% threshold method for vehicle detection
range_max = max(max(dataOut_test));  % find the maximal value
range_min = min(min(dataOut_test));  % minimal measurement

% change threshold

T_thres = range_max;
dT_thres = 0.05;
num_det = [];

while T_thres >= range_min

    % index of measuremnt above threshold
    PIR1_above_thres_index = PIR1_filtered >= T_thres;
    PIR2_above_thres_index = PIR2_filtered >= T_thres;
    PIR3_above_thres_index = PIR3_filtered >= T_thres;

    % OR combination of the three PIR sensors
    above_thres_index = PIR1_above_thres_index | PIR2_above_thres_index | PIR3_above_thres_index;

    circ_thres_index = circshift(above_thres_index,-1); % shift one bit

    det_index = circ_thres_index - above_thres_index;   % the index of detection, 1 begin-1, -1 end
    det_begin_index = find(det_index==1)+1;
    det_end_index = find(det_index==-1);
    % num of misclassified 
    num_mis = sum(det_begin_index(2:end)-det_end_index(1:end-1) <= 50);
    num_det = [num_det sum(det_index==1)-num_mis];
    
    % plot the most reasonable results
    if  sum(det_index==1)-num_mis == 324
        det_begin_index = find(det_index==1)+1;
        det_end_index = find(det_index==-1);
        
        figure
        plot(PIR1_filtered,'b');
        hold on
        plot(PIR2_filtered,'g');
        plot(PIR3_filtered,'r');
        scatter(find(PIR1_above_thres_index==1),...
            PIR1_filtered(PIR1_above_thres_index),'k.');
        scatter(find(PIR2_above_thres_index==1),...
            PIR2_filtered(PIR2_above_thres_index),'k.');
        scatter(find(PIR3_above_thres_index==1),...
            PIR3_filtered(PIR3_above_thres_index),'k.');
        plot([1, 71896],[T_thres, T_thres], 'k--', 'LineWidth',2);
        hold off
        title('323 detection');
    end
    
     % plot the most reasonable results
    if  sum(det_index==1)-num_mis == 363
        det_begin_index = find(det_index==1)+1;
        det_end_index = find(det_index==-1);
        
        figure
        plot(PIR1_filtered,'b');
        hold on
        plot(PIR2_filtered,'g');
        plot(PIR3_filtered,'r');
        scatter(find(PIR1_above_thres_index==1),...
            PIR1_filtered(PIR1_above_thres_index),'k.');
        scatter(find(PIR2_above_thres_index==1),...
            PIR2_filtered(PIR2_above_thres_index),'k.');
        scatter(find(PIR3_above_thres_index==1),...
            PIR3_filtered(PIR3_above_thres_index),'k.');
        plot([1, 71896],[T_thres, T_thres], 'k--', 'LineWidth',2);
        hold off
        title('363 detection');
    end
    
    
    T_thres = T_thres - dT_thres;


end