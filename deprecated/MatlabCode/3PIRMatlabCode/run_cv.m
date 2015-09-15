clear

profile on
format long g;
cv = cross_validation();

% Based on previous result
%w = 10; % window size
%s = ceil(w/5);
%[total_FP, total_FN]= cv.run_detection_cv(w, s, 'logit');



% speed estimation
% output_table = cv.run_several_detection_models();
[rms, pred_speed, true_speed] =...
    cv.run_speed_cv('zeroPadConv', 'row', 'halfThreshold', '1/t');

% remove row 165. That is when Rafi went to check the sensor
pred_speed(165) = [];
true_speed(165) = [];
% 
% 
% % plot the speed estimation result
% [s_true_speed, i_true_speed] = sort(true_speed);
% sorted_speed = [s_true_speed, pred_speed(i_true_speed)];
% 
% figure
% plot(sorted_speed(:,1),'b','LineWidth',2);
% hold on
% plot(sorted_speed(:,2), 'r','LineWidth',2);
% hold off
% legend('true speed', 'estimated speed');
% xlabel('Samples','FontSize',18);
% ylabel('Speed (mph)','FontSize',18);
% title('Visualization of speed estimation','FontSize',16);
%       
% 
% % plot in the original order
% figure
% plot(true_speed,'b','LineWidth',2);
% hold on
% plot(pred_speed,'r','LineWidth',2);
% hold off
% 
% plot in semi-original order; manipulate the matrix
m_true_speed = reshape(true_speed, 32, 9);
m_pred_speed = reshape(pred_speed, 32, 9);
ss_true_speed = [];
ss_pred_speed = [];
for i=1:8
    
    ss_true_speed = [ss_true_speed;...
        reshape( m_true_speed(4*(i-1)+1:4*i,:),36,1)];
    ss_pred_speed = [ss_pred_speed;...
        reshape( m_pred_speed(4*(i-1)+1:4*i,:),36,1)];
    
end

% % plot semi-sorted
figure
plot(ss_true_speed,'b','LineWidth',2);
hold on
plot(ss_pred_speed, 'r','LineWidth',2);
hold off
legend('true speed', 'estimated speed');
xlabel('Samples','FontSize',18);
ylabel('Speed (mph)','FontSize',18);
title('Visualization of speed estimation','FontSize',16);


profile viewer
















