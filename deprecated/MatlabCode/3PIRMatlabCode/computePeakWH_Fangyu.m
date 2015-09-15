function [peakW,peakH] = computePeakWH_Fangyu(rawSIG,freq,thr,amp)
%_______________________________________________________________________________
% <FUNCTION> 	Return the width and height of a single passing curve.
% <INPUT> 		rawSIG: a temperature vector
%				freq: frequency of the sampling
%				thr: threshold used to obtain height and width
%				amp: amplification factor
% <OUTPUT> 		peakW: width of the passing curve
%				peakH: height of the passing curve
%_______________________________________________________________________________

% By default, FREQUENCY = 12, THRESHOLD = 0.095
l0 = length(rawSIG);
ampSIG = resample(rawSIG,amp,1);
freq = freq * amp;
l = length(ampSIG);
ascSIG = sort(ampSIG); 


% Find ambient temperature
ambT = mean(ascSIG(1:floor(0.1*l)));
% Find height 
peakH = ascSIG(l) - ambT; % peakH represents the strength of the signal


% Find width
thrH = (peakH) * thr;

% THR(1:l) = thrH;
% trimSIG = ascSIG - THR;
% thrX = min(abs(trimSIG));
% i = find(trimSIG == thrX);
% if isempty(i)
% 	i = find(trimSIG == (0-thrX));
% end
% peakW = (l - i)/freq; % peakW in unit sec

% Find the width of the signal
peakW = sum(ampSIG>=thrH)/freq;


% % Plot the final results as well as the intermediate results
% figure
% %plot(	amp*(0:l0-1)/freq,rawSIG,'*', ...
% %		(1:l)/freq,ampSIG,'o')
%     %, ...
% 	%	(1:l)/freq,ascSIG,'o', ...
% 	%	(1:l)/freq,trimSIG,'o')
% scatter((1:l)/freq,ampSIG,'.g');
% hold on
% scatter(amp*(0:l0-1)/freq,rawSIG,'*');
% plot( [1, l/freq], [thrH, thrH], 'r','LineWidth',2);  % the threshold
% xlabel('Timestamp (sec)')
% ylabel('Signal strength (dimensionless)')
% legend('Raw signal','Upsampled signal', 'Threshold')
% %, ...
% %'Sorted signal in ascending order', ...
% %'Trimed signal by threshold')
% title('Find the peak and width')





