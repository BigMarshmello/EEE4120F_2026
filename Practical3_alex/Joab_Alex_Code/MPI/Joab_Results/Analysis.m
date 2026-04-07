% Define the matrix sizes you want to process based on your files
matrix_sizes = [4,5,6,7,8,9,10]; 

% Define the specific core numbers you want to analyze
target_cores = [1, 2, 4, 8, 12];

fig_Tcomp = figure('Name', 'Computation');
hold on
title('Computation Time vs Number of Cores for MPI Implementation');
xlabel('Number of cores (n)');
ylabel('Computation time (s)');
grid on;

% Initialize the Speedup figure
fig_speedup = figure('Name', 'Speedup');
hold on;
title('Speedup vs Number of Cores for MPI Implementation');
xlabel('Number of Cores (np)');
ylabel('Speedup (S = T_1 / T_p)');
grid on;

% Initialize the Speedup figure
fig_speedup_Total = figure('Name', 'Speedup_total');
hold on;
title('Speedup for Computation + Initialisation vs Number of Cores for MPI Implementation');
xlabel('Number of Cores (np)');
ylabel('Speedup (S = T_1 / T_p)');
grid on;

% Initialize the Efficiency figure
fig_eff = figure('Name', 'Efficiency');
hold on;
title('Efficiency for Computation + Initialisation vs Number of Cores for MPI Implementation');
xlabel('Number of Cores (np)');
ylabel('Efficiency (E = S / p)');
grid on;

% Loop through each dataset
for i = 1:length(matrix_sizes)
    size_val = matrix_sizes(i);
    filename = sprintf('benchmark_energy%d.csv', size_val);
    
    % Check if file exists before trying to read it
    if ~isfile(filename)
        warning('File %s does not exist. Skipping...', filename);
        continue;
    end
    
    % Read the CSV file into a table
    data = readtable(filename);
    
    % Filter to only include the target cores that actually exist in the file
    available_cores = unique(data.np);
    cores = intersect(target_cores, available_cores);
    num_cores = length(cores);
    
    if num_cores == 0
        warning('None of the target cores were found in %s.', filename);
        continue;
    end
    
    %avg_time = zeros(num_cores, 1);
    %avg_time_init = zeros(num_cores,1);
    %avg_time_total = zeros(num_cores,1);

    avg_time = zeros(numel(target_cores), 1);
    avg_time_init = zeros(numel(target_cores),1);
    avg_time_total = zeros(numel(target_cores),1);
    
    % Calculate average time for each targeted core count over the 100 iterations
    for c = 1:numel(target_cores)
        core_count = target_cores(c);
        %core_count = c;

        % Extract computation times for the current core count
        times_comp = data.Tcomp_seconds(data.np == core_count);
        times_init = data.Tinit_seconds(data.np == core_count);
        times_total = times_comp+times_init;
        
        % Average the iterations
        avg_time(c) = mean(times_comp);
        avg_time_init(c) = mean(times_init);
        avg_time_total(c) = mean(times_total);
    
    end
    
    % Base sequential time is the time taken by 1 core
    % Ensure 1 core data exists for proper scaling
    if cores(1) ~= 1
        warning('Core 1 data is missing in %s. Speedup might be calculated incorrectly.', filename);
    end
    T1 = avg_time(1);
    T1_init = avg_time_init(1);
    T1_total = avg_time_total(1);

    
    % Calculate Speedup and Efficiency
    speedup = T1 ./ avg_time;
    efficiency = speedup ./ target_cores;

    speedup_total = T1_total ./ avg_time_total;
    efficiency_total = speedup_total ./ target_cores';

    %speedup_total
    %target_cores'

    %numel(target_cores)
    %numel(avg_time)
    %plot computation time vs threads
    figure(fig_Tcomp);
    plot(target_cores,avg_time, '-o', 'LineWidth', 2, 'DisplayName', sprintf('%dx%d Matrix', size_val, size_val));
    
    % Plot Speedup for this matrix size
    figure(fig_speedup);
    plot(target_cores, speedup, '-o', 'LineWidth', 2, 'DisplayName', sprintf('%dx%d Matrix', size_val, size_val));

    % Plot Speedup for this matrix size for the total speedup
    figure(fig_speedup_Total);
    plot(target_cores, speedup_total, '-o', 'LineWidth', 2, 'DisplayName', sprintf('%dx%d Matrix', size_val, size_val));
    
    % Plot Efficiency for this matrix size
    figure(fig_eff);
    plot(target_cores, efficiency_total, '-s', 'LineWidth', 2, 'DisplayName', sprintf('%dx%d Matrix', size_val, size_val));
end

figure(fig_Tcomp);
xticks(target_cores);
legend('Location','northwest');
hold off;

% --- Finalize Formatting for Speedup Plot ---
figure(fig_speedup);
% Add an Ideal Speedup line matching the target cores
plot(target_cores, target_cores, '--k', 'LineWidth', 1.5, 'DisplayName', 'Ideal Speedup');
% Fix the x-axis ticks to only show our target cores
xticks(target_cores); 
legend('Location', 'northwest');
hold off;

figure(fig_speedup_Total);
plot(target_cores, target_cores, '--k', 'LineWidth', 1.5, 'DisplayName', 'Ideal Speedup');
xticks(target_cores);
legend('Location','northwest');
hold off;

% --- Finalize Formatting for Efficiency Plot ---
figure(fig_eff);
% Add an Ideal Efficiency line (Efficiency = 1.0)
plot(target_cores, ones(size(target_cores)), '--k', 'LineWidth', 1.5, 'DisplayName', 'Ideal Efficiency');
% Fix the x-axis ticks to only show our target cores
xticks(target_cores);
legend('Location', 'northeast');
ylim([0 1.2]); % Set Y-limits cleanly scaled
hold off;

Output = [target_cores' avg_time_init avg_time avg_time_total speedup speedup_total];
writematrix(Output, "ResultsTable.csv");