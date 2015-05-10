% read serial and plot in real time

% this is the new version, processing the data as
% counter, PIR1_meas1, PIR2_meas1, PIR3_meas1, ...
% PIR1_meas8, PIR2_meas8, PIR3_meas8, ambient.

clear

num_meas = 3;
total_col = 27; %time, count, 25 data
% Data_orig = zeros(0,total_col); % original data which have 900 (should be negative)
Data = zeros(0,total_col); 
DataBig = zeros(0,total_col);

% index of the PIR1 from 27 columns
% row vectorsd
index_PIR1 = ([0 0 kron(ones(1,8), [1 0 0] ) 0] == 1);
index_PIR2 = ([0 0 kron(ones(1,8), [0 1 0] ) 0] == 1);
index_PIR3 = ([0 0 kron(ones(1,8), [0 0 1] ) 0] == 1);


if ~isempty(instrfind)
    fclose(instrfind);
    delete(instrfind);
end

% clear ports
s = serial('COM8');
% s = serial('/dev/tty.usbserial-AM0211V4');
set(s,'BaudRate',115200);
fopen(s);

lineCounter = 1;
while 1
    tline = fscanf(s);
    
    tline
    
    if isempty(tline)
        pause(0.1);
        sprintf('waiting to connect port');
        continue
    end
    
    tic
    % parse the string
    % remove unnecessary characters
    tline(1:2) = [];   
    tline(end-1:end) = [];
    
    pcounter = str2double(tline(end-1:end));
    tline(end-1:end) = [];
    tline    %simply display the line to make sure all data are correct
    
    % parse to 3 measurements
    ns = numel(tline);
    strMeas = cellstr( reshape( ...
        [tline repmat(' ',1,ceil(ns/num_meas)*num_meas-ns)],...
        num_meas,[])')';
    
    % line to save
    % counter, data, last ambient temp
    Meas = str2double(strMeas);
    dataLine = [now, pcounter, Meas];
    
    if length(dataLine) ~= total_col
        % corrupted
        Data(lineCounter,1) = -1;
        %Data_orig(lineCounter,1) = -1;
    else
        Data(lineCounter,:) = dataLine;
%         Data_orig(lineCounter,:) = dataLine;    % save original data
        
        % change the values which is greater than 900 to negative
%         corrected_Meas = Meas;
%         corrected_Meas( Meas>= 900 ) = 900 - Meas(Meas>=900);
%         newDataLine = [dataLine(1:2), corrected_Meas];
%         Data(lineCounter,:) = newDataLine;
    end
    
    
    % plot data for three sensors 
    % only plot 800 length, around 80 s
    if lineCounter <= 100
        plot_start = 1;
        plot_end = lineCounter;
        num_pts = 8*lineCounter;
    else
        plot_start = lineCounter - 100+1;
        plot_end = lineCounter;
        num_pts = 8*100;
    end
    toc
    
    if ~isempty(Data)
        
        PIR1 = reshape(Data(plot_start:plot_end,index_PIR1)', num_pts, 1);
        PIR2 = reshape(Data(plot_start:plot_end,index_PIR2)', num_pts,1);
        PIR3 = reshape(Data(plot_start:plot_end,index_PIR3)', num_pts,1);
        
        avgPIR1 = mean(Data(plot_start:plot_end,index_PIR1),2);
        avgPIR2 = mean(Data(plot_start:plot_end,index_PIR2),2);
        avgPIR3 = mean(Data(plot_start:plot_end,index_PIR3),2);
        ambientTmp = Data(plot_start:plot_end, total_col);
        % ambientTmp = kron(Data(plot_start:plot_end, 26),ones(8,1));
        
        % replace the digit 9 here with negative
        
        
        figure(1)
%         plot((plot_start):plot_end, avgPIR1/10, 'r','LineWidth',3);
%         hold on
%         plot((plot_start):plot_end, avgPIR2/10, 'm','LineWidth',3);
%         plot((plot_start):plot_end, avgPIR3/10, 'b','LineWidth',3);
%         plot((plot_start):plot_end, ambientTmp/10, 'k','LineWidth',3);
        
        plot((plot_start-1)*8+1:plot_end*8, PIR1/10, 'r','LineWidth',3);
        hold on
        plot((plot_start-1)*8+1:plot_end*8, PIR2/10, 'm','LineWidth',3);
        plot((plot_start-1)*8+1:plot_end*8, PIR3/10, 'b','LineWidth',3);
        plot((plot_start-1)*8+1:plot_end*8, kron(ambientTmp'/10, ones(1,8)), 'k','LineWidth',3);
        grid on
        hold off
        xlim([(plot_start-1)*8+1, plot_end*8]);
        % ylim([-5, 10]);
        % axis auto
        legend('PIR1','PIR2','PIR3','Ambient');
        xlabel('time','FontSize',14);
        ylabel('Temperature (C)','FontSize',14);
        title('PIR test','FontSize',16);
        
    end
    
    % increment line counter
    lineCounter = lineCounter+1;
    
    if lineCounter == 375*30; %roughly every 5 min
        formatOut = 'yy_mm_dd_HH_MM_SS_FFF';
        name = datestr(now,formatOut);
        % 16 will give 0 error, but just make sure. 
        % each csv file will be 50k for 375*27
        dlmwrite(name, Data,'precision',18);     
        
        % save into data Big
        DataBig = [DataBig; Data];
        
        % reset Data
        Data = zeros(0,total_col);
        lineCounter = 1;
    end
    
    
end
% fprintf(s,'*IDN?')
% out = fscanf(s);
fclose(s)
delete(s)
clear s




