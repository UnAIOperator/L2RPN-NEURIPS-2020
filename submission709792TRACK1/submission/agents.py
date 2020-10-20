import os

import networkx as nx

from grid2op.Agent import BaseAgent
from grid2op.Reward import L2RPNReward

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

import math

loaded = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+ 'actions_array_useful.npz')
actions_array = np.transpose(loaded['actions_array_useful'])

class NoisyLayer(nn.Module):
    def __init__(self, in_dim, out_dim, params=None, is_noisy=False):
        super(NoisyLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = params
        self.is_noisy = is_noisy
        self.sigma_init = 0.5
        self.mu_w = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.mu_b = nn.Parameter(torch.Tensor(out_dim))
        self.sigma_w = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.sigma_b = nn.Parameter(torch.Tensor(out_dim))
        # Epsilon is not trainable
        self.register_buffer("eps_w", torch.Tensor(out_dim, in_dim))
        self.register_buffer("eps_b", torch.Tensor(out_dim))
        self.init_params()
        self.update_noise()

    def init_params(self):
        # Trainable params
        nn.init.uniform_(self.mu_w, -math.sqrt(1 / self.in_dim), math.sqrt(1 / self.in_dim))
        nn.init.uniform_(self.mu_b, -math.sqrt(1 / self.in_dim), math.sqrt(1 / self.in_dim))
        nn.init.constant_(self.sigma_w, self.sigma_init / math.sqrt(self.out_dim))
        nn.init.constant_(self.sigma_b, self.sigma_init / math.sqrt(self.out_dim))

    def update_noise(self):
        self.eps_w.copy_(self.factorize_noise(self.out_dim).ger(self.factorize_noise(self.in_dim)))
        self.eps_b.copy_(self.factorize_noise(self.out_dim))

    def factorize_noise(self, size):
        # Modify scale to amplify or reduce noise
        x = torch.Tensor(np.random.normal(loc=0.0, scale=0.001, size=size))
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        return F.linear(x, self.mu_w + self.sigma_w * self.eps_w, self.mu_b + self.sigma_b * self.eps_b)



class DuelingDistributionalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, distr_params, is_noisy):
        super(DuelingDistributionalNetwork, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim, 400),
                                         nn.ReLU())
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_bins = distr_params["num_bins"]
        self.v_range = distr_params["v_range"]
        
        if is_noisy:
            self.advantage_layer_hidden = NoisyLayer(400, 400)
            self.advantage_layer_out = NoisyLayer(400, out_dim * self.num_bins)

            self.value_layer_hidden = NoisyLayer(400, 400)
            self.value_layer_out = NoisyLayer(400, self.num_bins)
        else:
            self.advantage_layer_hidden = nn.Linear(400, 400)
            self.advantage_layer_out = nn.Linear(400, out_dim * self.num_bins)

            self.value_layer_hidden = nn.Linear(400, 400)
            self.value_layer_out = nn.Linear(400, self.num_bins)

        self.advantage_act = nn.ReLU()
        self.value_layer_act = nn.ReLU()

        self.softmax = nn.Softmax(dim=2)

        self.threshold_powerFlow_safe = 0.95
        
    def update_noise(self):
        self.advantage_layer_hidden.update_noise()
        self.advantage_layer_out.update_noise()

        self.value_layer_hidden.update_noise()
        self.value_layer_out.update_noise()

    def action_distr(self, x):
        x = self.input_layer(x)

        # Advantage
        advantage = self.advantage_layer_hidden(x)
        advantage = self.advantage_act(advantage)

        # Value
        value = self.value_layer_hidden(x)
        value = self.value_layer_act(value)

        advantage = self.advantage_layer_out(advantage).reshape(-1, self.out_dim, self.num_bins)
        value = self.value_layer_out(value).reshape(-1, 1, self.num_bins)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.softmax(q).clamp(1e-5)

    def forward(self, x):
        x = torch.sum(self.action_distr(x) * self.v_range, dim=2)
        return x
    
    

class Rainbow_agent(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """
    def __init__(self, action_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.action_space = action_space
        self.device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
        self.obs_size = 1040
        self.act_size = 386
        self.start_epoch = 1
        
        self.distr_params = {"num_bins": 51, "v_min": -1.0, "v_max": 1.0}
        self.prioritized_params = {"a": 0.6, "b": 0.6, "eps": 1e-5}
        
        self.distr_params["v_range"] = torch.linspace(self.distr_params["v_min"],
                                              self.distr_params["v_max"],
                                              self.distr_params["num_bins"]).to(self.device)
        self.model = DuelingDistributionalNetwork(self.obs_size, self.act_size, self.distr_params,
                                                         True).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.load_weights()

        self.threshold_powerFlow_safe = 0.95
        
        
    def find_action(self, state):
        legal_action = self.model(self.process_state(self.get_observation_vector(state)))
        action_index = int(legal_action.cpu().argmax().detach().cpu())
        
        action_asclass = self.action_space({})
        action_asclass.from_vect(actions_array[action_index,:])
        obs_, reward_simu_, done_, infos  = state.simulate(action_asclass)
    
        if done_ or sum(((( obs_.rho - 1) )[obs_.rho > 1.0]))>0:
            reward_simu_ = self.est_reward_update(obs_, reward_simu_, done_)
            additional_action = 100
            policy_chosen_list = np.argsort(legal_action.cpu().detach().numpy())[-1: -additional_action-1: -1]
            
            action_asclass = [None]*additional_action
            reward_simu1 = [0]*additional_action
            for i in range(additional_action):
                action_asclass[i] = self.action_space({})
                action_asclass[i].from_vect(actions_array[policy_chosen_list[0][i],:])
                obs_0, reward_simu1[i], done_0, _  = state.simulate(action_asclass[i])
                reward_simu1[i] = self.est_reward_update(obs_0,reward_simu1[i],done_0)
                if (not done_0) and (sum(((( obs_0.rho - 1) )[obs_0.rho > 1.0]))==0):
                    action_index = policy_chosen_list[0][i]
                    break
            
            if np.max(reward_simu1)>reward_simu_:
                print('has danger!', np.max(reward_simu1), reward_simu_)
                action_index = policy_chosen_list[0][np.argmax([reward_simu1])] # origin
        
        next_action = action_index
        
        
        return next_action

    
    def get_observation_vector(self,OBS_0):
        connectivity_ = OBS_0.connectivity_matrix()
        graph = nx.from_numpy_matrix(connectivity_)
        pagerank_ = nx.pagerank(graph)
        betweenness_centrality_ = nx.betweenness_centrality(graph)
        degree_centrality_ = nx.degree_centrality(graph)
        pagerank = []
        betweenness_centrality = []
        degree_centrality = []
        
        for k in sorted(pagerank_.keys()):
            pagerank.append(pagerank_[k])
            betweenness_centrality.append(betweenness_centrality_[k])
            degree_centrality.append(degree_centrality_[k])
    
        graph_features =  np.hstack((
            pagerank,
            betweenness_centrality,
            degree_centrality,
        ))
        
        forcasted_obs = OBS_0.get_forecasted_inj()
        prod_p_f = forcasted_obs[0]
        #prod_v_f = forcasted_obs[1]
        load_p_f = forcasted_obs[2]
        load_q_f = forcasted_obs[3]

        numeric_features = np.hstack((prod_p_f, load_p_f, load_q_f))
        
        prod_p = OBS_0.prod_p
        prod_q = OBS_0.prod_q
        #prod_v = OBS_0.prod_v
        load_p = OBS_0.load_p
        load_q = OBS_0.load_q
        #load_v = OBS_0.load_v
        rho_margin = 1 - OBS_0.rho
        
        numeric_features = np.hstack((numeric_features, prod_p, prod_q, load_p, load_q, rho_margin))
        
        topo_vect = OBS_0.topo_vect-1
        line_status = OBS_0.line_status*1
        
        selected_features = np.hstack((numeric_features, graph_features, topo_vect, line_status))
        
        return selected_features



    def process_state(self, state):
        return torch.from_numpy(state).type(torch.float32).to(self.device)

    def est_reward_update(self,obs,rw,done): 
        
        state_obs = obs
        rw_0 = rw - 5 * sum(((( state_obs.rho - 0.9) )[
                        state_obs.rho > 0.9]))- 10 * sum(((( state_obs.rho - 1) )[
                        state_obs.rho > 1])) if not done else -200
        return rw_0


    def load_weights(self):
        checkpoint = torch.load(os.path.split(os.path.realpath(__file__))[0]+'/'+ 'ckpt_279.pth')

        self.model.load_state_dict(checkpoint['net'])  #load learnable parameters

        self.optimizer.load_state_dict(checkpoint['optimizer'])  # optimizer configs
        
        self.start_epoch = checkpoint['epoch']  # Episodes
        print('Loaded from previous model!')


    #we reconnect lines that were in maintenance or attacked when possible
    def reco_line(self,observation):
          # add the do nothing
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        if np.any(can_be_reco):
            actions = [self.action_space({"set_line_status": [(id_, +1)]}) for id_ in np.where(can_be_reco)[0]]
            action=actions[0]

            osb_simu, _reward, _done, _info = observation.simulate(action, time_step=0)
            if (np.all(osb_simu.rho < self.threshold_powerFlow_safe)) & (len(_info['exception'])==0):
                return action
        return None
                
    # we order overloads by usage rate but also by criticity giving remaining timesteps for overload before disconnect
    def getRankedOverloads(self, observation):
        #timestepsOverflowAllowed = self.observation_space.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED
        timestepsOverflowAllowed = 3

        sort_rho = -np.sort(-observation.rho)  # sort in descending order for positive values
        sort_indices = np.argsort(-observation.rho)
        ltc_list = [sort_indices[i] for i in range(len(sort_rho)) if sort_rho[i] >= 1]

        # now reprioritize ltc if critical or not
        ltc_critical = [l for l in ltc_list if (observation.timestep_overflow[l] == timestepsOverflowAllowed)]
        ltc_not_critical = [l for l in ltc_list if (observation.timestep_overflow[l] != timestepsOverflowAllowed)]

        ltc_list = ltc_critical + ltc_not_critical
        return ltc_list  
    
    
    def act(self, observation,reward, done=False):
        """
        By definition, all "greedy" agents are acting the same way. The only thing that can differentiate multiple
        agents is the actions that are tested.

        These actions are defined in the method :func:`._get_tested_action`. This :func:`.act` method implements the
        greedy logic: take the actions that maximizes the instantaneous reward on the simulated action.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controller / agent.

        """

        
        if min(observation.rho < 1.0): # seems 0.8 is the best
            this_action = self.action_space({})
        else:
            action = self.find_action(observation)
            action_asvector = actions_array[action,:]
            this_action = self.action_space({})
            this_action.from_vect(action_asvector)

        # or we try to reconnect a line if possible
        action = self.reco_line(observation)
        if action is not None:
            best_action=action
            #return action
        else:
            best_action=self.action_space({}) 

        #our approach
        obs_forecast, _, done, _ = observation.simulate(this_action)
        obs_forecast1, _, done1, _ = observation.simulate(best_action)
        if not done and not done1:
           #if obs_forecast.rho.max() > obs_forecast1.rho.max():
           if obs_forecast.rho.max() < obs_forecast1.rho.max():    
                #res=best_action
                best_action=this_action
        if not done and done1:
            best_action=this_action   

        #our approach
        if done and done1:
            action_space={}
            ltc_list = self.getRankedOverloads(observation)
            n_overloads = len(ltc_list)
            idx=ltc_list[0]
            if idx:
                if idx < 21 or idx == 53 or idx == 54 or idx == 55:
                    #new_line_status_array = np.zeros(observation.rho.shape)
                    #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                    action_saved=self.action_space({'set_bus': {'substations_id': [(21, [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0])]}, 'redispatch': [(2, -2.6), (3, -2.6), (13, 2.6), (16, 2.6)]})
                else:
                    new_line_status_array = np.zeros(observation.rho.shape)
                    new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                    action_saved=self.action_space({'set_bus': {'substations_id': [(21, [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0])]}, 'redispatch': [(2, 2.6), (3, 2.6), (13, -2.6), (16, -2.6)]})
                best_action=action_saved
            obs_forecast1, _, done1, _ = observation.simulate(best_action)
            if done1:
                action_space = {}
                ltc_list = self.getRankedOverloads(observation)
                idx=ltc_list[0]
                if idx:
                    sub_or = observation.line_or_to_subid[idx]
                    if sub_or < 16 or sub_or == 16:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(0, -0.3), (2, -0.3), (3, -0.3), (4, -0.3), (10, 0.3), (13, 0.3), (19, 0.3), (21, 0.3)]
                        best_action=self.action_space(action_space)
                    if sub_or > 16 and sub_or < 27:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(2, 0.3), (3, 0.3), (4, 0.3), (10, -0.3), (13, -0.3), (16, -0.3)]
                        #res=action_saved
                        best_action=self.action_space(action_space)
                    if sub_or > 27:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(2, 0.3), (3, 0.3), (4, 0.3), (19, -0.3), (20, -0.3), (21, -0.3)]
                        #res=action_saved
                        best_action=self.action_space(action_space)        

        
        #return this_action
        return best_action
