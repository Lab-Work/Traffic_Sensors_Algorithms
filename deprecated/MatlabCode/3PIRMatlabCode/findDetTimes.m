% This funciton takes in the binary sequence into the the times of
% detection
%========================================
% input: k_ori, n by 3 matrix [start_time, end_time, 0/1]
% oupput: m by 2: start_time, end_time


function det_times = findDetTimes(k_ori)


k = k_ori(:,3);

% minlength is the thresshold on one car. 
% For instance, one detection can only last 10 1's
% disable for now
minlength = inf;

if (~isrow(k) && ~iscolumn(k) )
    error('Check k dimension\n');
elseif (iscolumn(k))
    k = k';
end
    
if (~all(isnan(k)==0))
    randvalue = randn*10;
    while ~all(k~=randvalue)
        randvalue = randn*10;
    end
    k(isnan(k)) = randvalue;
    randvalue = floor(randvalue*100000)/100000;
end

%eliminate possible numerical error
k_tmp = round(k*100000)/100000; 
originalSize = size(k_tmp,2);

i = 1;
indicedGroupValue = [0 0 0];    %initial offset
while (1) 
    
    i=i+1;
    [values, ivalues, ik_tmp] = unique(k_tmp,'stable');
    
    % set the starting indice
    indicedGroupValue(i,1) = indicedGroupValue(i-1,2)+1;
    % set the value
    indicedGroupValue(i,3) = values(1,1);
    
    if (size(values,2) > 1) %not last
        if (ivalues(2,1) <= minlength)
            % in case that we prefer a minlength: e.g. each group
            % aggregate at most 5 elements.
            indicedGroupValue(i,2) = ivalues(2,1)-1+ indicedGroupValue(i-1,2);
            k_tmp(:,1:ivalues(2,1)-1) = [];
        elseif (ivalues(2,1) > minlength)
            indicedGroupValue(i,2) = minlength + indicedGroupValue(i-1,2);
            % remove those processed elements
            k_tmp(:,1:minlength) = [];
        end
        
    % is the last element, and smaller than minlength    
    elseif(originalSize-indicedGroupValue(i,1) <= minlength)    
        indicedGroupValue(i,2) = originalSize;
        break;
        
    else
        indicedGroupValue(i,2) = minlength + indicedGroupValue(i-1,2);
        % remove those processed elements
        k_tmp(:,1:minlength) = [];
    end
    
end

% remove the initial offset (first row)
indicedGroupValue(1,:) = [];

if (~all(isnan(k_ori)==0))
    indicedGroupValue(indicedGroupValue(:,3)==randvalue,3) = NaN;
end

tmp_index = (indicedGroupValue(:,3)==1);
car_det = indicedGroupValue(tmp_index,:);

det_times = zeros(size(car_det,1),2);
det_times(:,1) = k_ori(car_det(:,1),1);
det_times(:,2) = k_ori(car_det(:,2),2);








