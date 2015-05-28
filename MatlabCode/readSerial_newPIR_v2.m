% This file read the new PIR data from the serial

% The format of the new PIR data is:
% #C(T00)(T10)(T20)(T30)(T01)....(T37)(Ta)(Counter)*
% #C(08)(T18)(T28)(T38)(T09)....(T315)(Ta)(Counter)*
% (T00) and (Ta) are 6-digit temperature in K
% (Counter) is two-digit 00~99
% Each measurement is splited to two strings with the same counter
% #C308.27307.15306.01306.30307.37305.48305.75306.34305.79305.47304.91305.82305.45305.10305.15308.53305.73304.75305.55310.39305.89304.90306.03308.74304.66304.48306.15308.17304.59304.71336.99363.60300.5915*
% #C304.59304.33307.99312.61304.63304.06304.58305.45303.96304.43304.03305.71304.42304.51303.82305.35303.79304.73304.56305.90304.64304.23304.73366.54305.26304.76305.98322.78305.18304.64307.14314.40300.5915*

% test git

clear

% clear ports
% s = serial('/dev/tty.usbserial-AM0211V4');
% s = serial('COM8');de

% for windows
% port = 'COM8';
% for mac
% port = '/dev/tty.usbserial-AM0211V4';
port = '/dev/tty.usbmodemfd121';


newPIR_data = readSerial(port, 115200);

% save data every 5 minutes
% newPIR_data.readSaveNewPIR(5, 18, 35);
newPIR_data.readPIRFromArduino(18,35);


% following code is just for computing the mean and standard deviation
data = newPIR_data.all_data;

mean_temp = zeros(4,16);
std_temp = zeros(4,16);

figure(2)
hold on
cmap = hsv(64);  %# Creates a 6-by-3 set of colors from the HSV colormap
for i=1:4
    
    for j = 1:16

        tmp_array = squeeze(data(i,j,:));
    
        % plot(tmp_array,'Color',cmap(4*(i-1)+j,:));  
        plot(tmp_array,'Color',[rand, rand, rand]);  
    
        % compute mean and std
        mean_temp(i,j) = mean(tmp_array);
        std_temp(i,j) = std(tmp_array);
        
    end
end
    












