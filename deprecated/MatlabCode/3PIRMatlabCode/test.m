% test speedEst.m class

clear

% read all data.txt
all_data_num = dlmread('./Data/filteredAllData_num.csv');


%======================
% read labels from table
% labels = readtable('./Data/cleaned_labels.txt');
% 
% % generate a training data set
% formatOut = 'yyyy-mm-dd HH:MM:SS.FFF';   
% 
% velocity(:,1) = table2array(labels(:,2));
% % clear format error
% begin_time = table2cell(labels(:,3));
% begin_time(48) = strcat(begin_time(48),'.000');
% end_time = table2cell(labels(:,4));
% 
% % only use first 250
% train_label = [datenum(begin_time(1:290), formatOut) ...
%     datenum(end_time(1:290), formatOut) velocity(1:290)];


% read labels from numerical csv
labels = dlmread('./Data/cleaned_labels_num_v2_4.csv');
begin_time = labels(:,2);
end_time = labels(:,3);
velocity = labels(:,4);

train_label = [begin_time(1:290), end_time(1:290), velocity(1:290)];


% test data
% test_label = [datenum(begin_time(251:290), formatOut) ...
%    datenum(end_time(251:290), formatOut)];

% train data
SpeedEst = speedEst( all_data_num );

%function model = speedTrain(obj, train_label, useConv, useShift, usePeak)
model = SpeedEst.speedTrain( train_label, 'wrapConv', 'row', 'halfThreshold' );

% est_speed = SpeedEst.speedTest( test_label );

SpeedEst.evaluateTrain;

% plot estimated results
% figure
% plot(velocity(251:290),'b','LineWidth',2);
% hold on
% plot(est_speed,'r','LineWidth',2);
% hold off
% legend('true velocity', 'estimated velocity');
% title('Speed estimation', 'FontSize',16);




% for debugging
% [t1, t2] = SpeedEst.plotSignalInDetection(86);
% [t3, t4] = SpeedEst.shiftSignalFrom(t1, t2, -1, -1);
% SpeedEst.replaceDetectionLabel([t1, t2], [t3, t4]);





% For debugging
% prob_data = SpeedEst.extractMatrix(all_data_num,...
%     735918.498868947150,735918.498960770085);
% figure
% plot(prob_data(:,2),'b');
% hold on
% plot(prob_data(:,3),'r');
% plot(prob_data(:,4),'g');
% hold off

% cell_labels = table2cell(labels);
% cell_labels(100,3) = cellstr(datestr(prob_data(30,1),formatOut));
% table_labels = cell2table(cell_labels);
% writetable(table_labels, './Data/cleaned_labels.txt');



% for txt
% num_label = [SpeedEst.cleaned_train_label;...
%             datenum(begin_time(291:end), formatOut), ...
%     datenum(end_time(291:end), formatOut) velocity(291:end)];

%for csv
% num_label = [SpeedEst.cleaned_train_label;...
%             begin_time(291:end), ...
%             end_time(291:end), velocity(291:end)];
% 
% % names = table2cell(labels(:,1));
% % raphi 1; yanning 2; emmanuel 3; brian 4;
% % num_names = zeros(321,1);
% % num_names( strcmp(names, 'raphi') ) = 1;
% % num_names( strcmp(names, 'yanning') ) = 2;
% % num_names( strcmp(names, 'emmanuel') ) = 3;
% % num_names( strcmp(names, 'brian') ) = 4;
% % counts = table2array(labels(:,5));
% 
% num_names = labels(:,1);
% counts = labels(:,5);
% 
% num_labels = [num_names, num_label, counts];
% 
% dlmwrite('cleaned_labels_num_v2_4.csv', num_labels,'precision',18);
% 








