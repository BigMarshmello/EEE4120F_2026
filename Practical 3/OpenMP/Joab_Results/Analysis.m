% Define the matrix sizes you want to process based on your files
matrix_sizes = [4,5,6,7,8,9,10]; 

% Define the specific core numbers you want to analyze
target_cores = [1, 2, 4, 8, 12];

% Initialize the Speedup figure
fig_speedup = figure('Name', 'Speedup');
hold on;
title('Speedup vs Number of Cores for OpenMP Implementation');
xlabel('Number of Cores (np)');
ylabel('Speedup (S = T_1 / T_p)');
grid on;

% Initialize the Efficiency figure
fig_eff = figure('Name', 'Efficiency');
hold on;
title('Efficiency vs Number of Cores for OpenMP Implementation');
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
    
    avg_time = zeros(num_cores, 1);
    
    % Calculate average time for each targeted core count over the 100 iterations
    for c = 1:num_cores
        core_count = cores(c);
        
        % Extract computation times for the current core count
        times = data.Tcomp_seconds(data.np == core_count);
        
        % Average the iterations
        avg_time(c) = mean(times);
    end
    
    % Base sequential time is the time taken by 1 core
    % Ensure 1 core data exists for proper scaling
    if cores(1) ~= 1
        warning('Core 1 data is missing in %s. Speedup might be calculated incorrectly.', filename);
    end
    T1 = avg_time(1); 
    
    % Calculate Speedup and Efficiency
    speedup = T1 ./ avg_time;
    efficiency = speedup ./ cores;
    
    % Plot Speedup for this matrix size
    figure(fig_speedup);
    plot(cores, speedup, '-o', 'LineWidth', 2, 'DisplayName', sprintf('%dx%d Matrix', size_val, size_val));
    
    % Plot Efficiency for this matrix size
    figure(fig_eff);
    plot(cores, efficiency, '-s', 'LineWidth', 2, 'DisplayName', sprintf('%dx%d Matrix', size_val, size_val));
end

% --- Finalize Formatting for Speedup Plot ---
figure(fig_speedup);
% Add an Ideal Speedup line matching the target cores
plot(target_cores, target_cores, '--k', 'LineWidth', 1.5, 'DisplayName', 'Ideal Speedup');
% Fix the x-axis ticks to only show our target cores
xticks(target_cores); 
legend('Location', 'northwest');
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