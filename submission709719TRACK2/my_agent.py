'''
TEAM: UN_aiGridOperator

TEAM MEMBERS

Daniel Alejandro Gualteros Gualteros <dgualterosg@unal.edu.co>, student at Universidad Nacional de Colombia
Edgard Leonardo Castaneda Garcia <elcastanedag@unal.edu.co>, student at Universidad Nacional de Colombia
David Leonardo Alvarez Alvarez <dlalvareza@unal.edu.co>, Associate Postdoc at Universidad Nacional de Colombia
Ivan Felipe Bonilla Vargas, Enel Colombia<ivan.bonilla@enel.com>, Senior Engineer at Enel-Codensa
Sergio Raul Rivera Rodriguez <srriverar@unal.edu.co>, Associate Professor at Universidad Nacional de Colombia 

Approach: some functions from amarot (the simulation time is improved) plus Ideas from "Magic Power Grids"
(there is no need of score increase since they did not send to track 2) 
and expert knowledge actions from heuristic actions from simulations, final action decision when "done" with a congestion rule through redispatch
'''

import numpy as np
from grid2op.Agent import BaseAgent
import pandapower as pp
#from lightsim2grid import LightSimBackend
#backend = LightSimBackend()

class ReconnectAgent(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """
    def __init__(self, action_space, observation_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        
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
        
    def act(self, observation, reward, done):
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

        #change the numbers according to netwrok of track 2
        if max(observation.rho) > 1:
            idx = np.argmax(observation.rho)
            if idx == 144 and self.operationsequence[idx] == 0 and observation.time_before_cooldown_sub[55] == 0:  #and observation.line_status[39]
                action_space["set_bus"] = {}
                action_space["set_bus"]["lines_or_id"] = [(146, 2), (148, 2), (152, 2)]
        #change the numbers according to netwrok of track 2
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

        '''                
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
        '''    
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
        
        #obs_forecast, _, done, _ = observation.simulate(res)
        #obs_forecast1, _, done1, _ = observation.simulate(best_action)
        #if not done and not done1:
        #   if obs_forecast.rho.max() > obs_forecast1.rho.max():
        #        res=best_action
            
        #print("action we take is:")
        #print(res)
        #return res 

        #our approach
        obs_forecast, _, done, _ = observation.simulate(res)
        obs_forecast1, _, done1, _ = observation.simulate(best_action)
        if not done and not done1:
           #if obs_forecast.rho.max() > obs_forecast1.rho.max():
           if obs_forecast.rho.max() < obs_forecast1.rho.max():    
                #res=best_action
                best_action=res
        if not done and done1:
            best_action=res        
        #if done and not done1:
        #    res=best_action        
        
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
                    new_line_status_array = np.zeros(observation.rho.shape)
                    #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                    #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                    action_space["redispatch"] = [(6, -0.6), (10, -0.6), (13, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]
                    best_action=self.action_space(action_space)
                else:
                    new_line_status_array = np.zeros(observation.rho.shape)
                    #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                    #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                    action_space["redispatch"] = [(6, 0.6), (10, 0.6), (13, 0.6), (16, 0.6), (35, -0.6), (36, -0.6), (60, -0.6), (61, -0.6)]
                    #res=action_saved
                    best_action=self.action_space(action_space)
                obs_forecast1, _, done1, _ = observation.simulate(best_action)
                if done1:           
                    if sub_or < 36:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(2, -0.6), (11, -0.6), (13, -0.6), (16, -0.6), (36, 0.6), (37, 0.6), (60, 0.6), (61, 0.6)]
                        best_action=self.action_space(action_space)
                    else:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(2, 0.6), (11, 0.6), (13, 0.6), (16, 0.6), (36, -0.6), (37, -0.6), (60, -0.6), (61, -0.6)]
                        #res=action_saved
                        best_action=self.action_space(action_space)
                obs_forecast1, _, done1, _ = observation.simulate(best_action)        
                if done1:           
                    if sub_or < 36:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(2, -0.6), (3, -0.6), (6, -0.6), (8, -0.6), (56, 0.6), (58, 0.6), (60, 0.6), (61, 0.6)]
                        best_action=self.action_space(action_space)
                    else:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(2, 0.6), (3, 0.6), (6, 0.6), (8, 0.6), (56, -0.6), (58, -0.6), (60, -0.6), (61, -0.6)]
                        #res=action_saved
                        best_action=self.action_space(action_space) 
                obs_forecast1, _, done1, _ = observation.simulate(best_action) 
                if done1:
                    if sub_or < 36:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(6, -1), (10, -1), (13, -1), (16, -1), (35, 1), (36, 1), (60, 1), (61, 1)]
                        best_action=self.action_space(action_space)
                    else:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(6, 1), (10, 1), (13, 1), (16, 1), (35, -1), (36, -1), (60, -1), (61, -1)]
                        #res=action_saved
                        best_action=self.action_space(action_space)
                obs_forecast1, _, done1, _ = observation.simulate(best_action)
                if done1:           
                    if sub_or < 36:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(2, -1), (11, -1), (13, -1), (16, -1), (36, 1), (37, 1), (60, 1), (61, 1)]
                        best_action=self.action_space(action_space)
                    else:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(2, 1), (11, 1), (13, 1), (16, 1), (36, -1), (37, -1), (60, -1), (61, -1)]
                        #res=action_saved
                        best_action=self.action_space(action_space)
                obs_forecast1, _, done1, _ = observation.simulate(best_action)        
                if done1:           
                    if sub_or < 36:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(2, -1), (3, -1), (6, -1), (8, -1), (56, 1), (58, 1), (60, 1), (61, 1)]
                        best_action=self.action_space(action_space)
                    else:
                        new_line_status_array = np.zeros(observation.rho.shape)
                        #new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                        #action_saved=self.action_space({'redispatch': [(6, -0.6), (10, -0.6), (6, -0.6), (16, -0.6), (35, 0.6), (36, 0.6), (60, 0.6), (61, 0.6)]})
                        action_space["redispatch"] = [(2, 1), (3, 1), (6, 1), (8, 1), (56, -1), (58, -1), (60, -1), (61, -1)]
                        #res=action_saved
                        best_action=self.action_space(action_space)               
                          

            
        print("action we take is:")
        print(res)
        #return res
        return best_action 

        #return res

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

def make_agent(env, submission_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your sudmission directory and return a valid agent.
    """
    res = ReconnectAgent(env.action_space, env.observation_space)
    return res
