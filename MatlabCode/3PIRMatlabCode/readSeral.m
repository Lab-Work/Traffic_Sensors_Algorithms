% read serial and plot in real time
clear

num_meas = 3;
total_col = 27; %time, count, 25 data
Data = zeros(0,total_col); 
len_plot = 800;


if ~isempty(instrfind)
    fclose(instrfind);
    delete(instrfind);
end

% clear ports
s = serial('/dev/tty.usbserial-AM0211V4');
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
    else
        Data(lineCounter,:) = dataLine;
    end
    
    
    % plot data for three sensors 
    % only plot 800 length, around 80 s
    if lineCounter <=100
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
        
        PIR1 = reshape(Data(plot_start:plot_end,3:10)', num_pts, 1);
        PIR2 = reshape(Data(plot_start:plot_end,11:18)',num_pts,1);
        PIR3 = reshape(Data(plot_start:plot_end,19:26)',num_pts,1);
        
        avgPIR1 = mean(Data(plot_start:plot_end,3:10),2);
        avgPIR2 = mean(Data(plot_start:plot_end,11:18),2);
        avgPIR3 = mean(Data(plot_start:plot_end,19:26),2);
        ambientTmp = Data(plot_start:plot_end, total_col);
        % ambientTmp = kron(Data(plot_start:plot_end, 26),ones(8,1));
        
        figure(1)
        plot((plot_start):plot_end, avgPIR1/10, 'r','LineWidth',3);
        hold on
        plot((plot_start):plot_end, avgPIR2/10, 'm','LineWidth',3);
        plot((plot_start):plot_end, avgPIR3/10, 'b','LineWidth',3);
        plot((plot_start):plot_end, ambientTmp/10, 'k','LineWidth',3);
        
%         plot((plot_start-1)*8+1:plot_end*8, PIR1/10, 'r','LineWidth',3);
%         hold on
%         plot((plot_start-1)*8+1:plot_end*8, PIR2/10, 'm','LineWidth',3);
%         plot((plot_start-1)*8+1:plot_end*8, PIR3/10, 'b','LineWidth',3);
%         plot((plot_start-1)*8+1:plot_end*8, ambientTmp/10, 'k','LineWidth',3);
        grid on
        hold off
        ylim([5, 20]);
        legend('PIR1','PIR2','PIR3','Ambient');
        xlabel('Sample count','FontSize',14);
        ylabel('Temperature (C)','FontSize',14);
        title('PIR test','FontSize',16);
        
    end
    
    % increment line counter
    lineCounter = lineCounter+1;
    
    if lineCounter == 375; %roughly every 5 min
        formatOut = 'mm_dd_HH_MM';
        name = datestr(now,formatOut);
        dlmwrite(name, Data,'precision',13);
        
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




