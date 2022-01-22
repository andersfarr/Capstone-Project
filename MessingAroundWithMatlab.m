mu = 100;
sigma = 30;
y = random('Normal',mu,sigma,1,10000);
order = 0;
inventory = 100;
total_cost = 0;
i = 1;
costs = zeros(1,365);

%State space for X - values taken 
possible_states = [0:1:200];

%Action state space - i.e. what's possible to order

possible_actions = [0:1:200];

generateQLearningPolicy(possible_states,possible_actions,y)

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

function gamma = generateQLearningPolicy(x_states, u_states, random_demand)
    maxItr = 10000;
    q = zeros(length(x_states), length(u_states));
    for i=1:maxItr
        
        % Starting from start position
        alpha_i = 1/i;
        demand = random_demand(i);
        
        q_next = zeros(length(x_states), length(u_states));
        int_demand = cast(demand,'int32');
        for x=1:length(x_states)
            for u=1:length(u_states)
                 X_next = x+u-int_demand;
                 if(X_next<= 0)
                    X_next = 1;
                 elseif(X_next > 200)
                    X_next = 200;
                 end
                 q_next(x,u) = (1-alpha_i)*q(x,u)+ alpha_i*(cost_func(u,x,demand) + 0.4*min(q(X_next)) - q(x,u));
            end
        end
        q = q_next;
    end
    [min_vals,gamma] = min(q,[], 2);
    gamma
end

%%Code found online - Q learning algorithm

