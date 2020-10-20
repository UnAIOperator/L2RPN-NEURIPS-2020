'''
TEAM: UN_aiGridOperator (Universidad Nacional de Colombia)
Approach: some functions from amarot (the simulation time is improved) 
plus Ideas from TONYS and expert knowledge actions from heuristic actions from simulations, 
final action decision when "done" with congestion rules through redispatch
'''
from grid2op.Agent import BaseAgent
from grid2op.Reward import BaseReward

import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch

N_TIME_S = 0
N_TIME_E = 5
N_GEN_P_S = 6
N_GEN_P_E = 68
N_GEN_QV_S = 68
N_GEN_QV_E = 192
N_LOAD_P_S = 192
N_LOAD_P_E = 291
N_LOAD_QV_S = 291
N_LOAD_QV_E = 489
N_LINE_OR_P_S = 489
N_LINE_OR_P_E = 675
N_LINE_OR_QVA_S = 675
N_LINE_OR_QVA_E = 1233
N_LINE_EX_P_S = 1233
N_LINE_EX_P_E = 1419
N_LINE_EX_QVA_S = 1419
N_LINE_OR_QVA_E = 1977
N_RHO_S = 1977
N_RHO_E = 2163
N_LINE_STAT_S = 2163
N_LINE_STAT_E = 2349
N_TINE_OF_S = 2349
N_TINE_OF_E = 2535
N_TOPO_S = 2535
N_TOPO_E = 3068
N_CD_LINE_S = 3068
N_CD_LINE_E = 3254
N_CD_SUB_S = 3254
N_CD_SUB_E = 3372
N_TIME_MANT_S = 3372
N_TIME_MANT_E = 3558
N_TIME_DURA_S = 3558
N_TIME_DURA_E = 3744
N_TAR_DISP_S = 3744
N_TAR_DISP_E = 3806
N_ACTL_DISP_S = 3806
N_ACTL_DISP_E = 3869

n_actions = 166
n_features = 719

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(n_features, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
    


class MyAgent(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a daughter of this
    class.
    """
    def __init__(self, action_space,net_dict,effective_topo,sub_info, observation_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.policy_net = Network()
        self.policy_net.load_state_dict(net_dict)
        self.policy_net.eval()
        self.effective_topo = effective_topo
        self.sub_info = sub_info

        #super().__init__(action_space)
        #self.name = name
        #self.grid = gridName  # IEEE14,IEEE118_R2 (WCCI or Neurips Track Robustness), IEEE118
        #logging.info("the grid you indicated to the Expert System is:" + gridName)
        self.curr_iter = 0
        self.sub_2nodes = set()
        self.lines_disconnected = set()
        self.action_space = action_space
        self.observation_space = observation_space
        self.threshold_powerFlow_safe = 0.95
        self.maxOverloadsAtATime = 3  # We should not run it more than
        self.config = {
            "totalnumberofsimulatedtopos": 25,
            "numberofsimulatedtopospernode": 5,
            "maxUnusedLines": 2,
            "ratioToReconsiderFlowDirection": 0.75,
            "ratioToKeepLoop": 0.25,
            "ThersholdMinPowerOfLoop": 0.1,
            "ThresholdReportOfLine": 0.2
        }
        self.reward_type = "MinMargin_reward"  # "MinMargin_reward"#we use the L2RPN reward to score the topologies, not the interal alphadeesp score
        
        self.nline = 186
        self.ngen = 22
        self.controllablegen = {0, 2, 3, 4, 10, 13, 16, 19, 20, 21}
        self.redispatchable = np.bool([True, False, True, True, True, False, False, False, False, False, True,
                               False, False, True, False, False, True, False, False, True, True, True])
        self.timestep = 0
        self.target_dispatch = np.zeros(self.ngen)
        self.base_power = [48, 28.2, 0, 150, 50, 0, 0, 0, 0, 0, 47.6, 0, 0, 70, 0, 0, 98.9, 0, 0, 300, 51, 180]
        self.lines_attacked = [0, 9, 13, 14, 18, 23, 27, 39, 45, 56]
        self.lines_cut = set()
        self.thermal_limits = [60.9, 231.9, 272.6, 212.8, 749.2, 332.4, 348., 414.4, 310.1,
                              371.4, 401.2, 124.3, 298.5, 86.4, 213.9, 160.8, 112.2, 291.4,
                              489., 489., 124.6, 196.7, 191.9, 238.4, 174.2, 105.6, 143.7,
                              293.4, 288.9, 107.7, 415.5, 148.2, 124.2, 154.4, 85.9, 106.5,
                              142., 124., 130.2, 86.2, 278.1, 182., 592.1, 173.1, 249.8,
                              441., 344.2, 722.8, 494.6, 494.6, 196.7, 151.8, 263.4, 364.1, 327.]
        self.operationsequence = np.zeros(self.nline)
        self.recoversequence = np.zeros(self.nline)

        self.action_space = action_space
        #self.observation_space = observation_space
        self.threshold_powerFlow_safe = 0.95

        
    def get_state(self,observation):
        obs_as_vect = observation.to_vect()
        states = np.append(obs_as_vect[N_RHO_S:N_RHO_E],obs_as_vect[N_TOPO_S:N_TOPO_E]) # rho & topo
        return states
        
    def emergency_monitor(self,obs):
        has_overflow = False
        obs_as_vect = obs.to_vect()
        rho = obs_as_vect[N_RHO_S:N_RHO_E]
        if np.amax(rho)>=1.0:
            has_overflow = True
        return has_overflow 
    
    def normal_operation(self, obs, N_actions):
        # if there is a line disconnected and cooldown is OK, then reconnect it
        line_status = obs.line_status
        line_CD = obs.time_before_cooldown_line
        reconnected_id = -1
        
        grid2op_action = self.action_space({}) # do nothing as baseline
        action_idx = N_actions-1
        
        if 0 in line_status: # if there is a line disconnected
            # detect which line is disconnected and cooldown is OK
            disconnected_line_id = np.asarray(np.where(line_status == 0))[0]
            for i in range(np.size(disconnected_line_id)):
                line_id = disconnected_line_id[i]
                if line_CD[line_id]==0:
                    reconnected_id = disconnected_line_id[i]
                    break
                
            if reconnected_id != -1: # all disconnected line in cooldown
                grid2op_action = self.action_space({"set_line_status": [(reconnected_id, 1)]})
        
        return action_idx, grid2op_action   
    
    def act(self, observation, reward, done):
        if self.emergency_monitor(observation):
            # cal Q_val
            state = torch.FloatTensor([self.get_state(observation)])
            Q_val = self.policy_net(Variable(state, requires_grad=True).type(torch.FloatTensor)).data
            Q_val = Q_val.detach().numpy()
            Q_sorted = np.argsort(-Q_val)[0]
            # action_prd = self.policy_net(Variable(state, requires_grad=True).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)
            action_buff = Q_sorted[0:50] # select top Q_vals index
            
            obs_as_vect = observation.to_vect()
            line_status = obs_as_vect[N_LINE_STAT_S:N_LINE_STAT_E]
            line_CD = obs_as_vect[N_CD_LINE_S:N_CD_LINE_E]
            sub_CD = obs_as_vect[N_CD_SUB_S:N_CD_SUB_E]
            topo = obs_as_vect[N_TOPO_S:N_TOPO_E]
            reconnected_id = -1
            
            max_rw = -1.0
            action_selected = self.action_space({}) # do nothing as baseline
            for i in range (0,len(action_buff)):
                action_index = action_buff[i]
                if 0 in line_status: # if there is a line disconnected
                    disconnected_line_id = np.asarray(np.where(line_status == 0))[0]
                    for i in range(np.size(disconnected_line_id)):
                        line_id = disconnected_line_id[i]
                        if line_CD[line_id]==0:
                            reconnected_id = disconnected_line_id[i]
                            break
                    if reconnected_id == -1: # line in cooldown time
                        if action_index<n_actions-1: # re-config
                            sub_id = self.effective_topo[action_index,533]
                            idx_node_start = self.sub_info[sub_id,1]
                            idx_node_end = self.sub_info[sub_id,1]+self.sub_info[sub_id,0]
                            sub_topo = topo[idx_node_start:idx_node_end]
                            if sub_CD[sub_id] != 0 or (-1 in sub_topo):# sub in CD or node disconnected
                                action = self.action_space({})
                            else:
                                target_topo = self.effective_topo[action_index,idx_node_start:idx_node_end]
                                action = self.action_space({"set_bus": {"substations_id": [(sub_id, target_topo)]}})
                        else:# do nothing
                            action = self.action_space({})# do nothing is selected 
                    else: # reconnect line
                        # print("reconnecting transmission line:",reconnected_id)
                        if action_index<n_actions-1: # re-config + reconnect
                            sub_id = self.effective_topo[action_index,533]
                            idx_node_start = self.sub_info[sub_id,1]
                            idx_node_end = self.sub_info[sub_id,1]+self.sub_info[sub_id,0]
                            sub_topo = topo[idx_node_start:idx_node_end]
                            if sub_CD[sub_id] != 0 or (-1 in sub_topo):# sub in CD or node disconnected
                                action = self.action_space({ "set_line_status": [(reconnected_id, 1)],
                                                            "set_bus": {"lines_or_id": [(reconnected_id, 1)],
                                                                        "lines_ex_id": [(reconnected_id, 1)]}})
                            else: # re-config + reconnect
                                target_topo = self.effective_topo[action_index,idx_node_start:idx_node_end]
                                action = self.action_space({ "set_line_status": [(reconnected_id, 1)],
                                                            "set_bus": {"lines_or_id": [(reconnected_id, 1)],
                                                                        "lines_ex_id": [(reconnected_id, 1)],
                                                                        "substations_id": [(sub_id, target_topo)]}})
                        else: # do nothing is selected, just reconnect
                            action = self.action_space({"set_line_status": [(reconnected_id, 1)],
                                                       "set_bus": {"lines_or_id": [(reconnected_id, 1)],
                                                                   "lines_ex_id": [(reconnected_id, 1)]}})
                else: # no line disconnected
                    if action_index<n_actions-1: # re-config
                        sub_id = self.effective_topo[action_index,533]
                        idx_node_start = self.sub_info[sub_id,1]
                        idx_node_end = self.sub_info[sub_id,1]+self.sub_info[sub_id,0]
                        sub_topo = topo[idx_node_start:idx_node_end]
                        if sub_CD[sub_id] != 0 or (-1 in sub_topo):# sub in CD or node disconnected
                                action = self.action_space({})
                        else:
                            target_topo = self.effective_topo[action_index,idx_node_start:idx_node_end]
                            action = self.action_space({"set_bus": {"substations_id": [(sub_id, target_topo)]}})
                    else: # do nothing is selected 
                        action = self.action_space({})# do nothing
                        
                obs_sim,rw_sim,done_sim,info_sim = observation.simulate(action)
                
                if not done_sim and rw_sim > max_rw:
                    max_rw = rw_sim
                    action_selected = action
                    
        # no overflow
        else: 
            _, action_selected = self.normal_operation(observation,n_actions)
        
        #return action_selected

        """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
        self.curr_iter += 1

        # Look for overloads and rank them
        ltc_list = self.getRankedOverloads(observation)
        counterTestedOverloads = 0

        n_overloads = len(ltc_list)
        if n_overloads == 0:  # if no overloads

            if (len(self.sub_2nodes) != 0):  # if any substation is not in the original topology, we will try to get back to it if it is safe
                for sub_id in self.sub_2nodes:
                    action = self.recover_reference_topology(observation, sub_id)
                    if action is not None:
                        best_action=action
                        #return action
            # or we try to reconnect a line if possible
            action = self.reco_line(observation)
            if action is not None:
                best_action=action
                #return action
            else:
                best_action=self.action_space({}) 
        else:
            best_action=self.action_space({})         
                
        action_space = {}
        self.timestep += 1

        if max(observation.rho) > 1:
            idx = np.argmax(observation.rho)
            if idx == 144 and self.operationsequence[idx] == 0 and observation.time_before_cooldown_sub[55] == 0:  #and observation.line_status[39]
                action_space["set_bus"] = {}
                action_space["set_bus"]["lines_or_id"] = [(146, 2), (148, 2), (152, 2)]

        lineidx = -1
        new_line_status_array = np.zeros(observation.rho.shape)
        if not observation.line_status.all():
            line_disconnected = np.where(observation.line_status == False)[0]
            for idx in line_disconnected[::-1]:
                if observation.time_before_cooldown_line[idx] > 0:
                    if idx == 160 or idx == 180:
                        if self.operationsequence[idx] == 1 and max(observation.rho) > 1:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(142, 2)]
                            action_space["set_bus"]["generators_id"] = [(25, 2)]
                            action_space["set_bus"]["loads_id"] = [(45, 2)]
                            action_space["redispatch"] = [(26, 2.8), (29, 2.8), (35, -2.8), (37, -2.8)]
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                            self.operationsequence[idx] += 1
                        if self.operationsequence[idx] == 0 and max(observation.rho) > 1:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(157, 2), (181, 2)]
                            action_space["set_bus"]["generators_id"] = [(32, 2)]
                            action_space["redispatch"] = [(26, 2.8), (29, 2.8), (35, -2.8), (37, -2.8)]
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                            self.operationsequence[idx] += 1
                    #if idx == 0 or idx == 9 or idx == 13 or idx == 14 or idx == 18 or idx == 23 or idx == 27 or idx == 39:
                    if idx == 23 or idx == 27: 
                        #we try to reconnect a line if possible
                        action_space=self.reco_line(observation)
                        if action_space is not None:
                            self.operationsequence[idx] += 1
                            return action_space
                        else:
                            return self.action_space({})        
            for idx in line_disconnected[::-1]:
                if observation.time_before_cooldown_line[idx] == 0:
                    lineidx = idx
                    print("Reconnect #", lineidx)
                    break

        for idx in range(self.nline):
            if observation.time_before_cooldown_line[idx] == 0:
                if idx == 160 or idx == 180:
                    if self.operationsequence[idx] == 1:
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(157, 1), (181, 1)]
                        action_space["set_bus"]["generators_id"] = [(32, 1)]
                        action_space["redispatch"] = [(26, -2.8), (29, -2.8), (35, 2.8), (37, 2.8)]
                        print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                        self.operationsequence[idx] -= 1
                    if self.operationsequence[idx] == 2:
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(142, 1)]
                        action_space["set_bus"]["generators_id"] = [(25, 1)]
                        action_space["set_bus"]["loads_id"] = [(45, 1)]
                        action_space["redispatch"] = [(26, -2.8), (29, -2.8), (35, 2.8), (37, 2.8)]
                        print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                        self.operationsequence[idx] -= 1
                    #if idx == 0 or idx == 9 or idx == 13 or idx == 14 or idx == 18 or idx == 23 or idx == 27 or idx == 39:
                    if idx == 23 or idx == 27: 
                        #we try to reconnect a line if possible
                        action_space=self.reco_line(observation)
                        if action_space is not None:
                            self.operationsequence[idx] -= 1
                            return action_space
                        else:
                            return self.action_space({})                 

        if lineidx != -1:
            new_line_status_array[lineidx] = 1
            action_space["set_line_status"] = new_line_status_array
            obs_forecast, _, done, _ = observation.simulate(self.action_space(action_space))
            if not done and obs_forecast.rho.max() < observation.rho.max():
                pass
            else:
                return self.action_space({})

        res = self.action_space(action_space)
        assert res.is_ambiguous()

        
        obs_forecast, _, done, _ = observation.simulate(res)
        obs_forecast1, _, done1, _ = observation.simulate(best_action)
        if not done and not done1:
           if obs_forecast.rho.max() > obs_forecast1.rho.max():
                res=best_action
            
        print("action we take is:")
        print(res)

        obs_forecast, _, done, _ = observation.simulate(res)
        obs_forecast1, _, done1, _ = observation.simulate(action_selected)
        if not done and not done1:
           if obs_forecast.rho.max() > obs_forecast1.rho.max():
                res=action_selected
        #if not done and done1:
        if done and not done1:
            res=action_selected
        #our approach
        if done and done1:
            action_space = {}
            ltc_list = self.getRankedOverloads(observation)
        #    n_overloads = len(ltc_list)
        #    if n_overloads == 0:
        #        best_action=res
        #    else:    
            idx=ltc_list[0]
            if idx:
                sub_or = observation.line_or_to_subid[idx]
                if sub_or < 36:
                    #new_line_status_array = np.zeros(observation.rho.shape)
                    #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                    #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                    action_space["set_bus"] = {}
                    action_space["set_bus"]["lines_ex_id"] = [(142, 2)]
                    action_space["set_bus"]["generators_id"] = [(25, 2)]
                    action_space["set_bus"]["loads_id"] = [(45, 2)]
                    action_space["redispatch"] = [(6, -0.6), (10, -0.6), (13, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]
                    best_action=self.action_space(action_space)
                else:
                    #new_line_status_array = np.zeros(observation.rho.shape)
                    #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                    #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                    action_space["set_bus"] = {}
                    action_space["set_bus"]["lines_ex_id"] = [(157, 2), (181, 2)]
                    action_space["set_bus"]["generators_id"] = [(32, 2)]
                    action_space["redispatch"] = [(6, 0.6), (10, 0.6), (13, 0.6), (16, 0.6), (35, -0.6), (36, -0.6), (60, -0.6), (61, -0.6)]
                    #res=action_saved
                    best_action=self.action_space(action_space)
                obs_forecast1, _, done1, _ = observation.simulate(best_action)
                if done1:           
                    if sub_or < 36:
                        #new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(142, 2)]
                        action_space["set_bus"]["generators_id"] = [(25, 2)]
                        action_space["set_bus"]["loads_id"] = [(45, 2)]
                        action_space["redispatch"] = [(2, -0.6), (11, -0.6), (13, -0.6), (16, -0.6), (36, 0.6), (37, 0.6), (60, 0.6), (61, 0.6)]
                        best_action=self.action_space(action_space)
                    else:
                        #new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(157, 2), (181, 2)]
                        action_space["set_bus"]["generators_id"] = [(32, 2)]
                        action_space["redispatch"] = [(2, 0.6), (11, 0.6), (13, 0.6), (16, 0.6), (36, -0.6), (37, -0.6), (60, -0.6), (61, -0.6)]
                        #res=action_saved
                        best_action=self.action_space(action_space)
                obs_forecast1, _, done1, _ = observation.simulate(best_action)        
                if done1:           
                    if sub_or < 36:
                        #new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(142, 2)]
                        action_space["set_bus"]["generators_id"] = [(25, 2)]
                        action_space["set_bus"]["loads_id"] = [(45, 2)]
                        action_space["redispatch"] = [(2, -0.6), (3, -0.6), (6, -0.6), (8, -0.6), (56, 0.6), (58, 0.6), (60, 0.6), (61, 0.6)]
                        best_action=self.action_space(action_space)
                    else:
                        #new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(157, 2), (181, 2)]
                        action_space["set_bus"]["generators_id"] = [(32, 2)]
                        action_space["redispatch"] = [(2, 0.6), (3, 0.6), (6, 0.6), (8, 0.6), (56, -0.6), (58, -0.6), (60, -0.6), (61, -0.6)]
                        #res=action_saved
                        best_action=self.action_space(action_space)
                obs_forecast1, _, done1, _ = observation.simulate(best_action)
                if done1:           
                    if sub_or < 36:
                        #new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(142, 2)]
                        action_space["set_bus"]["generators_id"] = [(25, 2)]
                        action_space["set_bus"]["loads_id"] = [(45, 2)]
                        action_space["redispatch"] = [(2, -0.2), (3, -0.2), (6, -0.2), (8, -0.2), (56, 0.2), (58, 0.2), (60, 0.2), (61, 0.2)]
                        best_action=self.action_space(action_space)
                    else:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(157, 2), (181, 2)]
                        action_space["set_bus"]["generators_id"] = [(32, 2)]
                        action_space["redispatch"] = [(2, 0.2), (3, 0.2), (6, 0.2), (8, 0.2), (56, -0.2), (58, -0.2), (60, -0.2), (61, -0.2)]
                        #res=action_saved
                        best_action=self.action_space(action_space)         
                obs_forecast1, _, done1, _ = observation.simulate(best_action)
                if done1:           
                    if sub_or < 36:
                        #new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(142, 2)]
                        action_space["set_bus"]["generators_id"] = [(25, 2)]
                        action_space["set_bus"]["loads_id"] = [(45, 2)]
                        action_space["redispatch"] = [(2, -0.2), (3, -0.2), (6, -0.2), (8, -0.2), (56, 0.2), (58, 0.2), (60, 0.2), (61, 0.2)]
                        best_action=self.action_space(action_space)
                    else:
                        #new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(157, 2), (181, 2)]
                        action_space["set_bus"]["generators_id"] = [(32, 2)]
                        action_space["redispatch"] = [(2, 0.2), (3, 0.2), (6, 0.2), (8, 0.2), (56, -0.2), (58, -0.2), (60, -0.2), (61, -0.2)]
                        #res=action_saved
                        best_action=self.action_space(action_space)         
                
                
                res=best_action                
        
        print("action we take is:")
        print(res)

        return res



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
        timestepsOverflowAllowed = self.observation_space.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED

        sort_rho = -np.sort(-observation.rho)  # sort in descending order for positive values
        sort_indices = np.argsort(-observation.rho)
        ltc_list = [sort_indices[i] for i in range(len(sort_rho)) if sort_rho[i] >= 1]

        # now reprioritize ltc if critical or not
        ltc_critical = [l for l in ltc_list if (observation.timestep_overflow[l] == timestepsOverflowAllowed)]
        ltc_not_critical = [l for l in ltc_list if (observation.timestep_overflow[l] != timestepsOverflowAllowed)]

        ltc_list = ltc_critical + ltc_not_critical
        return ltc_list  

    # for a substation we get the reference topology action
    def reference_topology_sub_action(self, observation, sub_id):
        topo_vec_sub = observation.state_of(substation_id=sub_id)['topo_vect']
        topo_target = list(np.ones(len(topo_vec_sub)))
        action_def = {"set_bus": {"substations_id": [(sub_id, topo_target)]}}
        action = self.action_space(action_def)
        return action

    def recover_reference_topology(self, observation, sub_id):
        topo_vec_sub = observation.state_of(substation_id=sub_id)['topo_vect']
        if (np.any(topo_vec_sub == 2)):
            action = self.reference_topology_sub_action(observation, sub_id)
            # we simulate the action to see if it is safe
            osb_simu, _reward, _done, _info = observation.simulate(action, time_step=0)

            if (np.all(osb_simu.rho < self.threshold_powerFlow_safe)) & (len(_info['exception']) == 0):
                self.sub_2nodes.discard(sub_id)
                return action

        return None    

class reward(BaseReward):
    """
    if you want to control the reward used by the envrionment when your agent is being assessed, you need
    to provide a class with that specific name that define the reward you want.

    It is important that this file has the exact name "reward" all lowercase, we apologize for the python convention :-/
    """
    def __init__(self):
        BaseReward.__init__(self)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous or is_done:
            rw = 0.0
        else:
            rw = 0.0
            obs = env.current_obs
            obs_as_vect = obs.to_vect()
            load_P = np.sum(obs_as_vect[192:291])
            gen_P = np.sum(obs_as_vect[6:68])
            line_status = obs_as_vect[2163:2349]
            rho = obs_as_vect[1977:2163]
            maintance_time = obs_as_vect[3372:3558]
            
            rw = load_P/gen_P
            for i in range(0,186):
                if line_status[i]==0 and maintance_time[i] != 0: rw += -0.1 # overflow disconnection
                elif rho[i] >= 1.0: rw += -0.05 # overflow
                
        return rw

def make_agent(env, submission_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your sudmission directory and return a valid agent.
    """
    effective_topo = np.load(os.path.join(submission_dir, "weights", "effective_topo.npy"))
    sub_info = np.load(os.path.join(submission_dir, "weights", "sub_info.npy"))
    net_dict = torch.load(os.path.join(submission_dir, "weights", "DQN_weights_0906.h5"),map_location='cpu')
    res = MyAgent(env.action_space,net_dict,effective_topo,sub_info,env.observation_space)
    return res

#def make_agent(env, submission_dir):
#    """
#    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
#    to your sudmission directory and return a valid agent.
#    """
#    res = ReconnectAgent(env.action_space, env.observation_space)
#    return res


