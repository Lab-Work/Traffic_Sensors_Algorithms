% readSerialClass
% Yanning Li
% This class read data from serial port
% different data string format will be parsed in different methods

classdef readSerial < handle
    
    properties
        % save all read data in the structure
        all_data;
        
        % time stampes associated with the data
        all_time;
        
        % all data with time
        % each row is one measurement
        all_measurement;
        
        % 5-min table
        data_to_save;
        
        % string, e.g. 'COM8'
        port;
        
        % 115200
        baudrate;
        
    end
    
    methods
        
        function obj = readSerial(port, baudrate)
            
            obj.port = port;
            obj.baudrate = baudrate;
            
            obj.all_data = [];
            
            obj.all_measurement = [];
            obj.data_to_save = [];
            
            
        end
        
        
        %==================================================================
        % The format of the new PIR data is:
        % #C(T00)(T10)(T20)(T30)(T01)....(T37)(Ta)(Counter)*
        % #C(08)(T18)(T28)(T38)(T09)....(T315)(Ta)(Counter)*
        % (T00) and (Ta) are 6-digit temperature in K
        % (Counter) is two-digit 00~99
        % Each measurement is splited to two strings with the same counter
        % #C308.27307.15306.01306.30307.37305.48305.75306.34305.79305.47304.91305.82305.45305.10305.15308.53305.73304.75305.55310.39305.89304.90306.03308.74304.66304.48306.15308.17304.59304.71336.99363.60300.5915*
        % #C304.59304.33307.99312.61304.63304.06304.58305.45303.96304.43304.03305.71304.42304.51303.82305.35303.79304.73304.56305.90304.64304.23304.73366.54305.26304.76305.98322.78305.18304.64307.14314.40300.5915*
        function readNewPIR(obj)
            
            timeStamp = [];
            
            s = serial(obj.port);
            
            set(s,'BaudRate',obj.baudrate);
            
            fopen(s);
            
            current_ID = 0;
            tmp_str_meas = '';
            
            
            loop_cnt = 1;
            while 1
                % start read data
                str_line = fscanf(s)
                
                % remove the end of line character
                str_line(end) = [];
                
                % the data length should be
                % 2+ 6*32 + 6 + 2 + 1
                if length(str_line) ~= 2+ 6*32 + 6 + 2 + 1
                    % read a new line
                    length(str_line)
                    continue
                end
                
                timeStamp = [timeStamp; now];
                
                % clean #C and *
                str_line(1:2) = [];
                str_line(end) = [];
                
                packetID = str2double(str_line(end-1:end));
                str_line(end-1:end) = [];
                
                [current_ID, packetID]
                if current_ID == packetID
                    % ambient temperature
                    T_a = str2double(str_line(end-5:end));
                    str_line(end-5:end) = [];
                    
                    % concatenate two string
                    str_meas = strcat(tmp_str_meas, str_line);
                    
                    length(str_meas)
                    
                    % send to str2matrix
                    meas_mat = obj.PIRstr2matrix(str_meas, 6);
                    
                    % save to all data
                    obj.all_data(:,:,loop_cnt) = meas_mat;
                    loop_cnt = loop_cnt+1;
                    
                    if ~isempty(meas_mat)
                        
                        % plot the result
                        disp(meas_mat)
                        
                        figure(1)
                        imagesc(meas_mat);
                        caxis([10,30]);
                        colorbar;
                        
                    end
                    
                    
                else
                    % remove ambient temp and save to tmp_str_meas
                    str_line(end-5:end) = [];
                    tmp_str_meas = str_line;
                    
                    % update current_ID
                    current_ID = packetID;
                end
                
                
                
            end
            
            
            
            
        end
        
        
        
        %==================================================================
        % same function as readNewPIR, but save data into files every 5
        % mins.
        % output: save data to file every minutes
        function readSaveNewPIR(obj, minutes, T_low, T_high)
            
            timeStamp = [];
            
            s = serial(obj.port);
            
            set(s,'BaudRate',obj.baudrate);
            
            fopen(s);
            
            current_ID = 0;
            tmp_str_meas = '';
            
            
            loop_cnt = 1;
            while 1
                % start read data
                str_line = fscanf(s)
                
                % remove the end of line character
                str_line(end) = [];
                
                % the data length should be
                % 2+ 6*32 + 6 + 2 + 1
                if length(str_line) ~= 2+ 6*32 + 6 + 2 + 1
                    % read a new line
                    length(str_line)
                    continue
                end
                
                cur_time = now;
                timeStamp = [timeStamp; cur_time];
                
                % clean #C and *
                str_line(1:2) = [];
                str_line(end) = [];
                
                packetID = str2double(str_line(end-1:end));
                str_line(end-1:end) = [];
                
                [current_ID, packetID]
                if current_ID == packetID
                    % ambient temperature
                    T_a = str2double(str_line(end-5:end));
                    T_a = T_a - 273.15; % convert to Celcius
                    str_line(end-5:end) = [];
                    
                    % concatenate two string
                    str_meas = strcat(tmp_str_meas, str_line);
                    
                    length(str_meas)
                    
                    % send to str2matrix
                    meas_mat = obj.PIRstr2matrix(str_meas, 6);
                    
                    % save to all data
                    obj.all_data(:,:,loop_cnt) = meas_mat;
                    loop_cnt = loop_cnt+1;
                    
                    if ~isempty(meas_mat)
                        
                        % plot the result
                        disp(meas_mat)
                        
                        figure(1)
                        imagesc(meas_mat);
                        caxis([T_low,T_high]);
                        colorbar;
                        
                        
                        % save the results into files
                        tmp_data = reshape(meas_mat', 1, 64);
                        obj.data_to_save = [obj.data_to_save;...
                                           [cur_time, tmp_data , T_a, current_ID]];
                        
                        % save every 1 min
                        if (cur_time - obj.data_to_save(1,1))*86400 >= minutes*60
                            formatOut = 'yy_mm_dd_HH_MM_SS_FFF';
                            name = datestr(now,formatOut);
                            dlmwrite(name, obj.data_to_save, 'precision',18); 
                            
                            % save to all_measure
                            
                            obj.all_measurement = [obj.all_measurement;...
                                                   obj.data_to_save];
                            
                            % clean data to save
                            obj.data_to_save = [];
 
                        end
                        
                        
                        
                        
                    end
                    
                    
                else
                    % remove ambient temp and save to tmp_str_meas
                    str_line(end-5:end) = [];
                    tmp_str_meas = str_line;
                    
                    % update current_ID
                    current_ID = packetID;
                end
                
                
                
            end
            
            
            
            
        end
        
        
        
        
        %==================================================================
        % Just to parse the sting to a matrix
        % The entire measurement 4 by 16
        function mat = PIRstr2matrix(obj, str, num_digit)
            
            len_str = length(str);
            
            % check length
            if len_str ~= 6*4*16
                disp('warning: incorrect string data format\n');
                mat = [];
                
                return
            end
            
            str_meas = cellstr( reshape( str, num_digit,[])')';
            
            mat = reshape( str2double(str_meas), 4, 16);
            
            % return in Celcius
            mat = mat-273.15;
            
        end
        
        
        
        
        %===========================
        % read an plot data collected using arduino
        function readPIRFromArduino(obj, T_min, T_max)
            
            timeStamp = [];
            
            s = serial(obj.port);
            
            set(s,'BaudRate',obj.baudrate);
            
            fopen(s);
            
            
            loop_cnt = 1;
            while loop_cnt <= 120
                % start read data
                str_line = fscanf(s);
                
                % wait until data comes
                if length(str_line) ~= 388
                    continue
                end
                
                % remove the end of line character
                % #23.23,.....,23.23,*
                str_line(1) = [];
                str_line(end-3:end) = [];
                
                obj.all_time = [obj.all_time; now];
                
                % convert to a matrix
                str_array = strsplit(str_line,',');
                temp_array = str2double(str_array);
                
                meas_mat = reshape(temp_array, 4, 16);
                    
                % save to all data
                obj.all_data(:,:,loop_cnt) = meas_mat;
                loop_cnt = loop_cnt +1;
                
                if ~isempty(meas_mat) && mod(loop_cnt,12)==0
                    
                    % plot the result
                    disp(meas_mat)
                    
                    figure(1)
                    imagesc(meas_mat);
                    caxis([T_min,T_max]);
                    colorbar;
                    
                end
                
                
            end
            
            
            
        end
        
        
        
        
    end
    
    
end