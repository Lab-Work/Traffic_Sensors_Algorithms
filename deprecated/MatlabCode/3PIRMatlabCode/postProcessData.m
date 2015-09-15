% post process data
% plot some basic figures

clear

% read all data.txt
allDatanum = dlmread('../Data/allData_unfiltered.csv');

PIR1_col = allDatanum(:,2)/10;
PIR2_col = allDatanum(:,3)/10;
PIR3_col = allDatanum(:,4)/10;

f = figure;
set(f, 'Units', 'normalized');
set(f, 'Position', [0 0 1 .5]);
plot(PIR1_col, 'r','LineWidth',3);
hold on
plot(PIR2_col, 'm','LineWidth',3);
plot(PIR3_col, 'b','LineWidth',3);
grid on
hold off
% ylim([-5, 10]);
axis auto
xlim([1,length(PIR1_col)]);
h = legend('PIR1','PIR2','PIR3');
set(h,'FontSize',16);
xlabel('Samples','FontSize',18);
ylabel('Temperature (C)','FontSize',18);
title('Unfiltered PIR sensor data','FontSize',20);


Y = fft(PIR1_col);
Y_mag = abs(Y);
% figure
% plot(Y_mag)
% num_bins = length(Y_mag);
% figure
% plot(0:1/(num_bins-1):1, Y_mag);


% High pass filter
hpFilt = designfilt('highpassiir','FilterOrder', 8, ...
         'PassbandFrequency', 1,'PassbandRipple',0.2, ...
         'SampleRate',81896);
% fvtool(hpFilt)
dataIn = [PIR1_col, PIR2_col, PIR3_col]; 
dataOut = filter(hpFilt,dataIn);

PIR1_filtered = dataOut(:,1);
PIR2_filtered = dataOut(:,2);
PIR3_filtered = dataOut(:,3);


f = figure;
set(f, 'Units', 'normalized');
set(f, 'Position', [0 0 1 .5]);
plot(PIR1_filtered(18850:18930), 'r','LineWidth',3);
hold on
plot(PIR2_filtered(18850:18930), 'm','LineWidth',3);
plot(PIR3_filtered(18850:18930), 'b','LineWidth',3);
grid on
hold off
% ylim([-5, 10]);
axis auto
% xlim([1,length(PIR1_col)]);
h = legend('PIR1','PIR2','PIR3');
set(h,'FontSize',16);
xlabel('Samples','FontSize',18);
ylabel('Temperature (C)','FontSize',18);
title('High pass filtered PIR data','FontSize',20);






% % band pass filter
% bpFilt = designfilt('bandpassiir','FilterOrder',20, ...
%          'HalfPowerFrequency1', 1,'HalfPowerFrequency2',81840, ...
%          'SampleRate',180e3);
% fvtool(bpFilt)
% dataIn = PIR1_col; 
% dataOut2 = filter(bpFilt,dataIn);
% 
% figure
% plot(PIR1_col/max(PIR1_col),'b');
% hold on
% plot(dataOut2/max(abs(dataOut2)),'r');
% hold off
% title('band pass 1 Hz ~ 80 kHz')



























