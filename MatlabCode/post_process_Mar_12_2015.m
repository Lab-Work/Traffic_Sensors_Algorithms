% process data
% index of the PIR1 from 27 columns
% row vectors

clear

allDataNov15 = dlmread('../Data/data_Mar12_2015.csv');

index_PIR1 = ([0 0 kron(ones(1,8), [1 0 0] ) 0] == 1);
index_PIR2 = ([0 0 kron(ones(1,8), [0 1 0] ) 0] == 1);
index_PIR3 = ([0 0 kron(ones(1,8), [0 0 1] ) 0] == 1);


% read allDataNov15.csv data
PIR1 = allDataNov15(:,index_PIR1);
PIR2 = allDataNov15(:,index_PIR2);
PIR3 = allDataNov15(:,index_PIR3);

% change time stamps to 10237 x 8 evenly divided stamps
% Note that the time stamp is when a line of data received.
% Hence -dt/8
timeStamps = zeros(size(allDataNov15,1),8);
timeStamps(:,1) = allDataNov15(:,1);
dt = (allDataNov15(2:end,1) - allDataNov15(1:end-1,1))/8;
dt = [dt(1); dt];   % assuming the first row has the same dt

for i = 2:8
    
    timeStamps(:,i) = timeStamps(:,i-1)-dt;
    
end

%flip matrix along column
times = flip(timeStamps,2);
times_col = reshape(times', numel(times),1);
PIR1_col = reshape(PIR1',numel(PIR1),1)/10;
PIR2_col = reshape(PIR2',numel(PIR2),1)/10;
PIR3_col = reshape(PIR3',numel(PIR3),1)/10;

allData = [times_col, PIR1_col, PIR2_col, PIR3_col];

dlmwrite('allData_num_Mar13_2015.csv', allData,'precision',18);
%
% save into file
formatOut = 'yy-mm-dd/HH:MM:SS.FFF';

strTime = datestr(times_col, formatOut);
% save in a table
T = table(strTime, times_col, PIR1_col, PIR2_col, PIR3_col);

writetable(T, 'allData.txt');


% here start to plot and save the images
% data to plot
% Data = allData(1:200,:);
% 
% while 1
%     
%     figure(1)
%     
%     plot((plot_start-1)*8+1:plot_end*8, PIR1/10, 'r','LineWidth',3);
%     hold on
%     plot((plot_start-1)*8+1:plot_end*8, PIR2/10, 'm','LineWidth',3);
%     plot((plot_start-1)*8+1:plot_end*8, PIR3/10, 'b','LineWidth',3);
%     plot((plot_start-1)*8+1:plot_end*8, kron(ambientTmp'/10, ones(1,8)), 'k','LineWidth',3);
%     grid on
%     hold off
%     % ylim([-5, 10]);
%     axis auto
%     legend('PIR1','PIR2','PIR3','Ambient');
%     xlabel('Sample count','FontSize',14);
%     ylabel('Temperature (C)','FontSize',14);
%     title('PIR test','FontSize',16);
%     
%     
%     
%     % increment line counter
%     lineCounter = lineCounter+1;
%     
%     if lineCounter == 375; %roughly every 5 min
%         formatOut = 'yy_mm_dd_HH_MM_SS_FFF';
%         name = datestr(now,formatOut);
%         % 16 will give 0 error, but just make sure.
%         % each csv file will be 50k for 375*27
%         dlmwrite(name, Data,'precision',18);
%         
%         % save into data Big
%         DataBig = [DataBig; Data];
%         
%         % reset Data
%         Data = zeros(0,total_col);
%         lineCounter = 1;
%     end
%     
%     
% end













