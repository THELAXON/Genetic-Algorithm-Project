map = im2bw(imread('random_map.bmp')); % Load Map
start = [1, 1]; % Preset start and finish values for the solution.
finish = [500, 500];

devisionScale = 20; % Choosing how much to deviate a particular point from a line so this does it by 20 pixels
noOfPointsInSolution = 20; % Number of points in the solution 
noOfiterations = 2000; % Max Number of iterations to be iterated through 
population_size = 100; % Number of chromosomes in population in each iteration

fittest = zeros(noOfiterations, 1); % Variable to store fitness value
population = zeros(population_size, 2 * noOfPointsInSolution); %stores the population required for genetic algorithm

obstacle_penalty_scale = 2500;% Adjustable penalty in fitness if black pixels are found the distance for the fitness increases meaning it isn't an optimal path 
uniform_range = 5; % Used to adjust the range of mutation for uniform mutation.
consecutive_fitness_counter = 0; % Counter for consecutive iterations with the same fitness

% Selection Prompts user to choose a selection method
disp('Selection Methods:')
disp('0 = Roulette Wheel Selection')
disp('1 = Tournament Selection')
disp('2 = Rank Based Selection')
selectionInput = input('Enter 0, 1 or 2 for the Selection method: ');

% Cross-over Prompts user to choose a cross-over for the genetic algorithm
disp('Cross-Over')
disp('0 = K-Point')
disp('1 = Uniform')
crossoverInput = input('Enter 0 or 1 for the Cross-over operator: ');

% Mutation Prompts user to choose the type of mutation they would like see
disp('Mutation')
disp('0 = Uniform Mutation')
disp('1 = Swap Mutation')
mutationInput = input('Enter 0 or 1 for the Mutation operator: ');


tic; % Timer starts here to elapse the time after user makes decisions

for i = 1:population_size % This loop is used to generate random coordinates for the no of solutions to be included in the bitmap
    [xCoordinates, yCoordinates] = generateCoordinates(start, finish, noOfPointsInSolution, devisionScale, map);
    population(i, 1:noOfPointsInSolution) = xCoordinates;
    population(i, noOfPointsInSolution + 1:noOfPointsInSolution * 2) = yCoordinates;
end

fitness_values = [zeros(population_size)];  % extra column is added to the end for fitness scores
population = [population, zeros(population_size, 1)];

for iteration = 1:noOfiterations            % Calculates the fitness for the current population
    for i = 1:population_size
        xCoordinates = population(i, 1:noOfPointsInSolution);
        yCoordinates = population(i, noOfPointsInSolution + 1:noOfPointsInSolution * 2);
        fitness = calcFitness(xCoordinates, yCoordinates, map, obstacle_penalty_scale);
        fitness_values(i) = fitness;
        population(i, 2 * noOfPointsInSolution + 1) = fitness;
    end
    population = sortrows(population, 2 * noOfPointsInSolution + 1);
    fittest(iteration, 1) = population(1, 2 * noOfPointsInSolution + 1); % save score of fittest in this generation k for plotting

    if iteration > 1 && fittest(iteration, 1) == fittest(iteration - 1, 1) % Checks if fitness values remain the same for 5 consecutive iterations
        consecutive_fitness_counter = consecutive_fitness_counter + 1; % Increments if it is the same
    else
        consecutive_fitness_counter = 0;
    end

    if consecutive_fitness_counter == 5                    %If it reaches 5 times then the optimal path has been found.
        disp('Terminated due to attaining fittest values.');
        break;
    end

    new_Population = zeros(population_size, 2 * noOfPointsInSolution); %Initialise new population for next generation of the solutions.
    new_Population(1:(0.1 * population_size), :) = population(1:(0.1 * population_size), 1:2 * noOfPointsInSolution);
    new_Population_num = (0.1 * population_size);

    % While looped until the generation of new population is full
    while (new_Population_num < population_size)
        % Selection based on user input
        if selectionInput == 0
            % Roulette selection 
            [parent_1, parent_2] = RouletteWheelSelection(population, noOfPointsInSolution);
        elseif selectionInput == 1
            % Tournament Selection
            [parent_1, parent_2] = TournamentSelection(2, population, noOfPointsInSolution);
        elseif selectionInput == 2
            % Rank Based Selection
            [parent_1, parent_2] = RankBasedSelection(population, noOfPointsInSolution);
        end


        % Crossover based on user input
        if crossoverInput == 0
            % K-Point
            [offspring_1, offspring_2] = k_Point_Crossover(parent_1, parent_2);
        elseif crossoverInput == 1
            % Uniform crossover
            [offspring_1, offspring_2] = uniform_Crossover(parent_1, parent_2);
        end

        new_Population_num = new_Population_num + 1;
        new_Population(new_Population_num, :) = offspring_1;
        if (new_Population_num < population_size)
            new_Population_num = new_Population_num + 1;
            new_Population(new_Population_num, :) = offspring_2;
        end

    end

    population(:, 1:2 * noOfPointsInSolution) = new_Population;
    % Mutation based on user input
    if mutationInput == 0
        % Uniform mutation
        mutation_rate = 0.5;
        population = uniformMutation(population, 0.1 * population_size, population_size, noOfPointsInSolution, uniform_range, mutation_rate);
    elseif mutationInput == 1
        % Swap mutation
        mutation_rate = 0.05;
        population = swapMutation(population, 0.1 * population_size, population_size, noOfPointsInSolution, mutation_rate);
    end
end

disp(['Elapsed Time: ', num2str(toc), ' seconds']); % Gives the elapsed time to find the optimal path
disp('Euclidean Distance of the optimal path :'); % Used to display the optimal path that is used fo the figure shown.
disp(population(1, 2 * noOfPointsInSolution + 1));
displayFigure(map, population(1, 1:noOfPointsInSolution), population(1, noOfPointsInSolution + 1:2 * noOfPointsInSolution));

function [offspring_1, offspring_2] = uniform_Crossover(parent_1, parent_2)
    if (rand < 0.6)
        crossoverMask = rand(size(parent_1)) < 0.5; % Randomise which genes to take from each parent

        offspring_1 = parent_1 .* crossoverMask + parent_2 .* (1 - crossoverMask);
        offspring_2 = parent_2 .* crossoverMask + parent_1 .* (1 - crossoverMask);
    else
        offspring_1 = parent_1;
        offspring_2 = parent_2;
    end
end

function [offspring_1, offspring_2] = k_Point_Crossover(parent_1, parent_2)
    % Generate k random crossover points
    k = 4;
    crossoverPoints = sort(randperm(length(parent_1), k));

    % children should initially have parents' genetic material in k-point
    offspring_1 = parent_1;
    offspring_2 = parent_2;

    if (rand < 0.6)
        % Perform crossover at the selected points
        for i = 1:2:length(crossoverPoints)
            startIdx = crossoverPoints(i);
            endIdx = min(crossoverPoints(i + 1), length(parent_1));

            % Swap genetic material between parents at the crossover points
            offspring_1(startIdx:endIdx) = parent_2(startIdx:endIdx);
            offspring_2(startIdx:endIdx) = parent_1(startIdx:endIdx);
        end
    end
end

function choice = Roulette(weights)

    accumulated = cumsum(weights);
    p = rand();
    chosen_index = -1;
    for index = 1:length(accumulated)
        if (accumulated(index) > p)
            chosen_index = index;
            break;
        end
    end
    choice = chosen_index;
end

function [parent_1, parent_2] = RouletteWheelSelection(population, noOfPointsInSolution)
    inverse_fitness = 1 ./ population(:, 2 * noOfPointsInSolution + 1);
    weights = inverse_fitness / sum(inverse_fitness);

    choice_1 = Roulette(weights);
    choice_2 = Roulette(weights);

    parent_1 = population(choice_1, 1:2 * noOfPointsInSolution);
    parent_2 = population(choice_2, 1:2 * noOfPointsInSolution);
end

function [parent_1, parent_2] = TournamentSelection(tournamentSize, population, noOfPointsInSolution)
    % Number of tournaments to run
    numTournaments = size(population, 1);

    parent_1 = zeros(1, 2 * noOfPointsInSolution);
    parent_2 = zeros(1, 2 * noOfPointsInSolution);

    for i = 1:numTournaments
        % Randomly choose individuals for the tournament
        tournamentIndices = randperm(numTournaments, tournamentSize);

        % Evaluate fitness for each participant in the tournament
        tournamentFitness = 1 ./ population(tournamentIndices, 2 * noOfPointsInSolution + 1);

        % Choose the index of the winner (the one with the highest fitness which is most optimal path)
        [~, winnerIndex] = max(tournamentFitness);

        % Assign the winner as one of the parents
        if i == 1
            parent_1 = population(tournamentIndices(winnerIndex), 1:2 * noOfPointsInSolution);
        else
            parent_2 = population(tournamentIndices(winnerIndex), 1:2 * noOfPointsInSolution);
        end
    end
end

function [parent_1, parent_2] = RankBasedSelection(population, noOfPointsInSolution)

    % Number of individuals in the population
    populationSize = size(population, 1);

    % Calculate ranks for each individual based on their fitness
    [~, sortedIndices] = sort(population(:, 2 * noOfPointsInSolution + 1));
    ranks = zeros(populationSize, 1);
    ranks(sortedIndices) = 1:populationSize;

    % Calculate selection probabilities based on inverse ranks
    selectionProbabilities = 1 ./ ranks;

    % Normalize probabilities to ensure they sum to 1
    selectionProbabilities = selectionProbabilities / sum(selectionProbabilities);

    % Select parents using random sampling based on probabilities
    selectedIndices = randsample(1:populationSize, 2, true, selectionProbabilities);

    % Assign selected parents
    parent_1 = population(selectedIndices(1), 1:2 * noOfPointsInSolution);
    parent_2 = population(selectedIndices(2), 1:2 * noOfPointsInSolution);
end

function population = uniformMutation(population, start_index, end_index, noOfPointsInSolution, uniform_range, mutation_rate)
    % Iterate through each individual in the population
    for i = start_index:end_index
        % Iterate through each gene in the chromosome
        for j = 1:2 * noOfPointsInSolution
            % Check if mutation occurs for this gene
            if rand() < mutation_rate
                % Add a small random value to the gene
                population(i, j) = population(i, j) + randi([-uniform_range, uniform_range]); % You can adjust the mutation strength
                if (population(i, j) < 1)
                    population(i, j) = 1;
                end
                if (population(i, j) > 500)
                    population(i, j) = 500;
                end
            end
        end
    end
end

function population = swapMutation(population, start_index, end_index, noOfPointsInSolution, mutation_rate)
    % Iterate through each individual in the population
    for i = start_index:end_index
        if rand() < mutation_rate
            mutationPoint = randi([1, noOfPointsInSolution]);
            temp = population(i, mutationPoint);
            population(i, mutationPoint) = population(i, mutationPoint + noOfPointsInSolution);
            population(i, mutationPoint + noOfPointsInSolution) = temp;
        end
    end
end
function displayFigure(map, xCoordinate, yCoordinate)
    map = cat(3, map, map, map);
    xCoordinate = [1, xCoordinate, 500];
    yCoordinate = [1, yCoordinate, 500];

    % Convert map to double
    map = double(map);

    % fin the number of points
    noOfPointsInSolution = numel(xCoordinate);

    % Set the pixels corresponding to the points to blue
    for i = 1:noOfPointsInSolution
        map(yCoordinate(i), xCoordinate(i), :) = [0, 0, 1]; % Dark Blue color for the co-ordinates
    end

    % Join the points with a blue line
    for i = 2:noOfPointsInSolution
        map = insertShape(map, 'Line', [xCoordinate(i-1), yCoordinate(i-1), xCoordinate(i), yCoordinate(i)], 'Color', 'blue');% Dark Blue color for the lines connecting the dots together.
    end

    figure;
    imshow(map);
    title('Most Optimal Path');
end

function [xCoordinates, yCoordinates] = generateCoordinates(start, finish, noOfPointsInSolution, devisionScale, map)
    noOfPointsInSolution = noOfPointsInSolution + 2;
    % Generate linearly spaced coordinates
    xCoordinates = linspace(start(1), finish(1), noOfPointsInSolution);
    yCoordinates = linspace(start(2), finish(2), noOfPointsInSolution);

    % Add random deviations to the intermediate coordinates
    xDeviation = devisionScale * randn(1, noOfPointsInSolution - 2);
    yDeviation = devisionScale * randn(1, noOfPointsInSolution - 2);

    xCoordinates(2:end-1) = round(xCoordinates(2:end-1) + xDeviation);
    yCoordinates(2:end-1) = round(yCoordinates(2:end-1) + yDeviation);

    % Clip coordinates to be within image boundaries
    imageSize = size(map);
    xCoordinates = max(1, min(imageSize(2), xCoordinates));
    yCoordinates = max(1, min(imageSize(1), yCoordinates));

    xCoordinates = xCoordinates(2:end-1);
    yCoordinates = yCoordinates(2:end-1);
end


function [x, y] = bresenham(x1, y1, x2, y2) % used to calculate the points along a line between two given points 
    dx = abs(x2 - x1);
    dy = abs(y2 - y1);

    if x1 < x2
        sx = 1;
    else
        sx = -1;
    end

    if y1 < y2
        sy = 1;
    else
        sy = -1;
    end

    err = dx - dy;

    x = zeros(1, max(dx, dy) + 1);
    y = zeros(1, max(dx, dy) + 1);

    k = 1; % Index for x and y arrays

    while true
        x(k) = x1;
        y(k) = y1;
        k = k + 1;

        if x1 == x2 && y1 == y2
            break;
        end

        e2 = 2 * err;

        if e2 > -dy
            err = err - dy;
            x1 = x1 + sx;
        end

        if e2 < dx
            err = err + dx;
            y1 = y1 + sy;
        end
    end

    x = x(1:k-1);
    y = y(1:k-1);
end



function fitness = calcFitness(xCoordinates, yCoordinates, map,obstacle_penalty_scale)
    xCoordinates = [1, xCoordinates, 500];
    yCoordinates = [1, yCoordinates, 500];

    blackPixelCounter = 0;% Keeps track of points that lie on black pixels in the binary image. The count of black pixels on the line contributes to the fitness calculation.
    noOfPointsInSolution = numel(xCoordinates);

    for i = 2:noOfPointsInSolution
        [x, y] = bresenham(xCoordinates(i-1), yCoordinates(i-1), xCoordinates(i), yCoordinates(i));% the purpose is to determine the points along the line connecting two consecutive coordinates in the solution path
        for j = 1:numel(x)
            if map(y(j), x(j), 1) == 0
                blackPixelCounter = blackPixelCounter + 1; % Adds one every time the path comes across a black pixel.
            end
        end
    end

    distances = sqrt(diff(xCoordinates).^2 + diff(yCoordinates).^2);

    % Calculate the sum of distances
    totalDistance = sum(distances);

    % Calculate fitness by using total distance and the penalty from getting next to a black pixel
    fitness = totalDistance + (obstacle_penalty_scale * blackPixelCounter);

end
