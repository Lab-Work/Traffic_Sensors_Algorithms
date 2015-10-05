% CEE491 project
% Yanning


clear

% check the data
tic
veh = unsup_vehDet();
toc

%=====================================================
% MAP
% calibration time 10 min
% streaming packet every 3 min
tic
veh.MAP_model(10, 5, 'point', 0.9, 'update')
toc

%=====================================================
% show the temperature distribution
% veh.plot_temp_distr([1,2,3]);
% veh.avg_data(25, 5);


%=====================================================
% streaming data; re-fit gmm upon new data
% save all data
% tic
% assume data streaming in very 30 s = 30*12 = 360 pts
% veh.gmm_model_1(360, 'point')
% toc


%=====================================================
% show the 3D cluster
% veh.gmm_fit(veh.raw_data,2)
% veh.gmm_cluster(veh.raw_data, 'plot')
% 




















