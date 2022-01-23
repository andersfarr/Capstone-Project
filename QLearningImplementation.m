%Load training and test datasets from files
%y_train = readmatrix('y.txt');
y_test = readmatrix('y_test.txt');

%State space for X - values taken  (indices)
%possible_states = [0:1:20];

%Action state space - i.e. what's possible to order (indices)
%possible_actions = [0:1:20];

%quantization - map each "bin" in the state space to a value in "values"
keys = possible_states;
%Differential size quantization
%values = cat(2,[0:15:75],[80:5:125]);
%values = cat(2, values, [140:15:200]);
%values = [0:10:200]

%Create map (dictionary) to map values and keys
dataMap = containers.Map(values,keys);


policy = [0,0,0,0,0,0,0,1,0,0,3,0,0,0,2,0,0,1,0,0,0,3,1,1,0,0,0,1,1,0,1,2,0,0,1,2,0,1,0,0,0,1,0,0,0,1,0,1,0,2,0,1,0,0,0,0,0,0,0,0,0,0,2,1,3,1,0,5,0,1,0,0,5,2,3,1,0,0,5,7,6,2,1,2,7,0,0,0,0,0,29,16,14,11,3,6,6,15,32,44,28,15,33,47,14,45,32,28,39,31,26,20,34,47,39,27,46,47,70,35,46,45,45,61,42,43,61,64,62,53,62,22,57,56,42,61,61,55,38,83,63,67,68,68,47,69,64,54,77,71,52,74,55,56,59,66,63,64,59,81,88,95,76,74,55,84,68,89,108,80,69,89,60,74,99,88,109,79,62,62,62,100,91,87,79,83,93,82,76,78,97,83,103,100,93,91,87,119,100]

%writematrix(policy, 'policies.txt', 'WriteMode', 'append')

order = 0;
inventory = 100;
total_cost = 0;
i = 1;
costs = zeros(1,365);

%Loop through test data to get yearly earnings if policy "policy" used
for demand=y_test
    inventory
    inventory_step = cast(inventory, 'int16')
    step_policy = policy(199-inventory_step)
    total_cost = total_cost + cost_func(step_policy,demand,inventory)
    if(inventory+step_policy <=demand)
        inventory = 0
    else
        inventory = inventory+step_policy-demand
    end
   
end
%Print out total accrued cost (negative of profits)
total_cost

%Cost function, takes into account whether demand > order or not
function cost = cost_func(ordered,demanded,have)
    holding_cost = 5;
    order_cost = 30;
    if(have+ordered< demanded)
        profit = 100 * (have+ordered);
    else
        profit = 100 * demanded;
    end
    cost = holding_cost*(have+ordered)+order_cost*(ordered)-profit;
end

%State space encoding for generated state
function encoded = encodeX(data)
    if(data < 80)
        encoded = (data - mod(data,15))/15;
    elseif(data < 130)
        encoded = (data - mod(data,5))/5 - 10;
    else
        encoded = (data - mod(data,15))/15 + 6;
    end
end

% Q-learning generator function
function gamma = generateQLearningPolicy(x_states, u_states, random_demand, map)
    %Iterations of Q-learning algo
    maxItr = 1000000;
    %Empty output matrix "Q"
    q = zeros(length(x_states), length(u_states));
    %Keep track of how often been in spot x,u in q, used for decay rate of
    %algorithm
    alpha_count = zeros(length(x_states),length(u_states));
    %Initial state (no inventory) 
    x = 1;
    %Loop over all iterations, generate Q matrix values
    for i=1:maxItr
        %Random policy u
        u = cast(rand*20,'int32')+1;
        %Increment alpha counter, generate current alpha value
        alpha_count(x,u) = alpha_count(x,u)+1;
        alpha_i = 1/alpha_count(x,u);
        %Use quantization map to get data values for x and u
        data_X = map(x-1);
        data_U = map(u-1);
        %Get demand from given random demand data
        demand = random_demand(i);
        
        q_next = zeros(length(x_states), length(u_states));
        int_demand = cast(demand,'int32');
        %Find out "next state" given demand, policy and current state
        X_next = data_X+data_U-int_demand;
        %Ensure state within quantization bounds
        if(X_next<= 0)
           X_next = 1;
        elseif(X_next > 200)
           X_next = 200;
        end
        %Use quantization to get index of next X
        X_next = encodeX(X_next)+1;
        %Q-learning function to generate next q value
        q_next(x,u) = (1-alpha_i)*q(x,u)+ alpha_i*(cost_func(data_U,data_X,demand) + 0.4*min(q(X_next)) - q(x,u));
        %Increment x and q matrix values before next iteration
        q(x,u) = q_next(x,u);
        x = X_next;
    end
    %Find minimum of array, giving policy
    [min_vals,gamma] = min(q,[], 2);
    %Decrement policy (1-based indexing)
    gamma = gamma-1;
end

