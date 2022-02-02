%Run the policy creation
policy = create_mmpp_policy()

writematrix(policy, 'policies.txt')

function policy = create_mmpp_policy()
    % Generate a MMPP process as the distribution
    n_runs = 100000000;  % Number of days
    markov = [0.9 0.05 0.05; 0.2 0.8 0; 0.2 0 0.8];  % P matrix
    periodicity = [0.5 0.75 1 1.75];  % Simulates a fiscal year with a spending spree in the 4th quarter (i.e. holidays)
    [y, quarter_t, z_t, n_runs] = mmpp_distribution(100, 25, periodicity, markov, n_runs);
    
    max_state = 500;  % Max inventory
    
    %State space for X - values taken  (matrix, each col is a set of states)
    x_states = combvec([0:5:max_state], [1 2 3 4], [0 1 -1]);

    %Action state space - i.e. policy actions
    u_states = [0:5:max_state];

    %Generate the policy
    policy_ind = generate_mmpp_QLearningPolicy(x_states, u_states, y, quarter_t, z_t, max_state, n_runs);
    policy = u_states(policy_ind)
end

function [distribution, quarter, z_t, n_days] = mmpp_distribution(lambda_0, lambda_1, periodicity, markov_mat, days)
    % Will explore a Markov-Modulated Poisson Process N(t) = N_0(t) +
    % N_E(t)
    % This is equivalent to having a base process and a Markovian 'event'
    % wich can be triggered to positively or negatively impact the
    % underlying process
    % Will set an N_0(t) distribution
    lambda_t = repelem(lambda_0 * periodicity, 90);
    factor = cast(days/length(lambda_t), 'int32');
    n_days = factor * 360;  % Make into a year with 360 days
    quarter = repmat(repelem([1:1:4], 90), 1,factor);
    N_0 = poissrnd(repmat(lambda_t,1,factor),1,n_days);
    % Will add a Markov 'event' process
    mc = dtmc(markov_mat);
    z_t = simulate(mc, n_days-1);
    
    % Create to a second process
    N_E = poissrnd(lambda_1,1,n_days);
    
    % Needs a mask to multiply the boolean values (Stupid MATLAB uses too
    % much memory otherwise in simple array multiplication)
    z_p = z_t == 2;
    z_n = z_t == 3;
    N_Et = zeros(size(N_E));
    N_Et(z_p) = N_E(z_p);
    N_Et(z_n) = -N_E(z_n);
    
    % Combine the resulting processes
    N = N_0 + N_Et;
    distribution = max(N, 0);  % Must be positive
end

%Cost function, takes into account whether demand > order or not
function cost = cost_func(ordered, demanded, have)
    holding_cost = 5;
    order_cost = 30;
    if((have+ordered)< demanded)
        revenue = 100 * (have+ordered);
    else
        revenue = 100 * demanded;
    end
    cost = (holding_cost*(have+ordered)+order_cost*(ordered))-revenue;
end

%State space encoding for generated state
function encoded = quantize(data)
%     if(data < 80)
%         encoded = (data - mod(data,15))/15;
%     elseif(data < 130)
%         encoded = (data - mod(data,5))/5 - 10;
%     else
%         encoded = (data - mod(data,15))/15 + 6;
%     end
    encoded = round( data / 5 ) * 5;
end

% Gets the index of a quantization value
function quantization_index = quantize_idx(value)
    quantization_index = round( value / 5 ) + 1;  % Index starts at 1
end

% Q-learning generator function
function gamma = generate_mmpp_QLearningPolicy(x_states, u_states, random_demand, time_states, event_states, max_state, n_runs)
    %Iterations of Q-learning algo
    maxItr = n_runs;
    %Empty output matrix "Q"
    % Hardcode in the fact that there are 4 quarters and 3 demand events
    % (high, normal, low)
    q = zeros(length(u_states), 4, 3, length(u_states));
    %Keep track of how often been in spot x,u in q, used for decay rate of
    %algorithm
    alpha_count = zeros(length(u_states), 4, 3,length(u_states));
    %Initial state (no inventory) 
    x_idx = 1;
    %Loop over all iterations, generate Q matrix values
    for i=1:maxItr
        %Random policy u
        x = x_states(1, x_idx);
        u = cast(rand*(max_state - x),'int32');  % Policy can't pick values that are too big
        u_idx = quantize_idx(u);
        %Increment alpha counter, generate current alpha value
        alpha_count(x_idx, time_states(i), event_states(i), u_idx) = alpha_count(x_idx, time_states(i), event_states(i), u_idx)+1;
        alpha_i = 1/alpha_count(x_idx,time_states(i), event_states(i),u_idx);
        %Get demand from given random demand data
        demand = cast(random_demand(i), 'int32');
        x_next = x+u-demand;
        %Ensure state within quantization bounds
        if(x_next<= 0)
           x_next = 0;
        end
        %Use quantization to get index of next X
        x_next_idx = quantize_idx(x_next);
        %Q-learning function to generate next q value
        q(x_idx,time_states(i), event_states(i),u_idx) = (1-alpha_i)*q(x_idx,time_states(i), event_states(i),u_idx)+ alpha_i*(cost_func(u,demand,x) + 0.4*min(q(x_next_idx,time_states(i), event_states(i))) - q(x_idx,time_states(i), event_states(i),u_idx));
        x_idx = x_next_idx;
    end
    %Find minimum of array, giving policy
    [min_vals,gamma] = min(q,[], 4);
end

