%Load training and test datasets from files
y_train = readmatrix('1st-OrderMC.txt');
y_train = y_train*10;
%y_test = readmatrix('y_test.txt');

%State space for X - values taken  (indices)
possible_states = [0:1:20];

%Action state space - i.e. what's possible to order (indices)
possible_actions = [0:1:20];

%quantization - map each "bin" in the state space to a value in "values"
keys = possible_states;
%Differential size quantization
%values = cat(2,[0:15:75],[80:5:125]);
%values = cat(2, values, [140:15:200]);
values = [0:10:200];

%Create map (dictionary) to map values and keys
dataMap = containers.Map(keys,values);
MC_order = 2;
%encodeX(ones(2)+1, 2);

policy = generateKthOrderQLearningPolicy(possible_states, possible_actions, y_train, dataMap, MC_order);
writematrix(policy, 'policies.txt', 'WriteMode', 'append')

order = 0;
inventory = 100;
total_cost = 0;
i = 1;
costs = zeros(1,365);

%Loop through test data to get yearly earnings if policy "policy" used
for demand=y_test
    inventory;
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
total_cost;

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
function encoded = quantize(data)
    %if(data < 80)
    %    encoded = (data - mod(data,15))/15;
    %elseif(data < 130)
    %    encoded = (data - mod(data,5))/5 - 10;
    %else
    %    encoded = (data - mod(data,15))/15 + 6;
    %end
    encoded = cast(data/10, 'int32');
end

function encoded = encodeX(data, k, num_states)
    encoded = 1;
    for i = 1:k
        encoded = encoded + (data(i)-1)*num_states^(k-i);
    end
end

% Q-learning generator function
function gamma = generateKthOrderQLearningPolicy(x_states, u_states, random_demand, map, k)
    %Iterations of Q-learning algo
    maxItr = 1000000;
    %Empty output matrix "Q"
    state_num = length(x_states);
    max_X = encodeX(ones(k)*length(x_states), k, state_num)
    q = zeros(max_X, length(u_states));
    %Keep track of how often been in spot x,u in q, used for decay rate of
    %algorithm
    alpha_count = zeros(max_X,length(u_states));
    %Initial state (no inventory for past k iterations) 
    x = ones(k);
    %Loop over all iterations, generate Q matrix values
    for i=1:maxItr
        %Random policy u
        u = cast(rand*20,'int32')+1;
        %Increment alpha counter, generate current alpha value
        x_encoded = encodeX(x,k,state_num);
        alpha_count(x_encoded,u) = alpha_count(x_encoded,u)+1;
        alpha_i = 1/alpha_count(x_encoded,u);
        %Use quantization map to get data values for x and u
        data_X = map(x(1)-1);
        data_U = map(u-1);
        %Get demand from given random demand data
        demand = random_demand(i);
        int_demand = cast(demand,'int32');
        %Find out "next state" given demand, policy and current state
        x_next = data_X+data_U-int_demand;
        %Ensure state within quantization bounds
        if(x_next<= 0)
           x_next = 1;
        elseif(x_next > 200)
           x_next = 200;
        end
        %Use quantization to get index of next X
        x_next =quantize(x_next)+1;
        x_next_vector = encodeX(cat(2, x_next, x(2:length(x))),k, state_num);
        %Q-learning function to generate next q value
        q_next = (1-alpha_i)*q(x_encoded,u)+ alpha_i*(cost_func(data_U,demand,data_X) + 0.4*min(q(x_next_vector)) - q(x_encoded,u));
        %Increment x and q matrix values before next iteration
        q(x_encoded,u) = q_next;
        x(2:length(x)) = x(1:length(x)-1);
        x(1) = x_next;
    end
    %Find minimum of array, giving policy
    [min_vals,gamma] = min(q,[], 2);
    %Decrement policy (1-based indexing)
    gamma = gamma-1;
end

