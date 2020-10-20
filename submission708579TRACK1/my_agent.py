'''
TEAM: UN_aiGridOperator

TEAM MEMBERS

Daniel Alejandro Gualteros Gualteros <dgualterosg@unal.edu.co>, student at Universidad Nacional de Colombia
Edgard Leonardo Castaneda Garcia <elcastanedag@unal.edu.co>, student at Universidad Nacional de Colombia
David Leonardo Alvarez Alvarez <dlalvareza@unal.edu.co>, Associate Postdoc at Universidad Nacional de Colombia
Ivan Felipe Bonilla Vargas, Enel Colombia<ivan.bonilla@enel.com>, Senior Engineer at Enel-Codensa
Sergio Raul Rivera Rodriguez <srriverar@unal.edu.co>, Associate Professor at Universidad Nacional de Colombia 

Approach some functions from amarot (the simulation time is improved) plus Ideas from "Magic Power Grids" (a score increase of more than 3 points) 
and expert knowledge actions from heuristic actions from simulations, final action decision when "done" with a congestion rule through redispatch
'''

import numpy as np
from grid2op.Agent import BaseAgent
import pandapower as pp
from copy import deepcopy

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
        
        self.nline = 59
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

        self.operationtimestep = np.zeros(self.nline)
        self.operationoption = np.zeros(self.nline)
        self.operationlineaction = np.ones(self.nline) * (-1)
        self.tooperateline = -1

        self.action_space = action_space
        #self.observation_space = observation_space
        self.threshold_powerFlow_safe = 0.95
        
    def act(self, observation, reward, done):
        """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
        self.curr_iter += 1

        ##some few lineas from amarot
        # Look for overloads and rank them
        ltc_list = self.getRankedOverloads(observation)
        counterTestedOverloads = 0

        n_overloads = len(ltc_list)
        #if n_overloads == 0:  # if no overloads

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
        #else:
        #    best_action=self.action_space({})         
                
        ## FROM MPG (submission 673575) combined with our heuristic rules (the score is improved from 22.5325 to 26.479944)        
        action_space = {}
        self.timestep += 1
        self.tooperateline = -1

        if np.any(self.operationtimestep > 0):
            idx = np.argmax(self.operationtimestep)
            action_space["set_bus"] = {}
            action_space["set_bus"]["lines_or_id"] = [(32, 1), (34, 1), (37, 1)]
            tmpaction = self.action_space(action_space)
            obs_, _, done, _ = observation.simulate(tmpaction)
            if idx == 31 and self.operationsequence[idx] == 1 and obs_.rho[31] < 0.8 and not done \
                    and observation.time_before_cooldown_sub[23] == 0:
                print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                self.tooperateline = 31
            else:
                action_space = {}

        if max(observation.rho) > 1:
            idx = np.argmax(observation.rho)
            if idx == 31 and self.operationsequence[idx] == 0 and observation.line_status[39] and observation.time_before_cooldown_sub[23] == 0:
                action_space["set_bus"] = {}
                action_space["set_bus"]["lines_or_id"] = [(32, 2), (34, 2), (37, 2)]
                print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                self.tooperateline = 31

        lineidx = -1
        
        new_line_status_array = np.zeros(observation.rho.shape)
        if not observation.line_status.all():
            line_disconnected = np.where(observation.line_status == False)[0]
            for idx in line_disconnected[::-1]:
                if observation.time_before_cooldown_line[idx] > 0:
                    if idx == 9:
                        self.action_space({'set_bus': {'substations_id': [(7, [1.0, 2.0, 2.0, 1.0, 2.0, 1.0])]}, 'redispatch': [(2, 2.7), (3, 2.7), (20, -2.7), (21, -2.7)]})
                    if idx == 13:
                        self.action_space({'set_bus': {'substations_id': [(13, [1.0, 1.0, 1.0, 2.0, 2.0])]}, 'redispatch': [(10, 2.7), (13, 2.7), (16, -2.7), (19, -2.7)]})
                    if idx == 18:    
                        self.action_space({'set_bus': {'substations_id': [(23, [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0])]}, 'redispatch': [(2, 2.7), (4, 2.7), (20, -2.7), (21, -2.7)]})
                    if idx == 39:
                        self.action_space({'set_bus': {'substations_id': [(21, [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0])]}, 'redispatch': [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]})
                    #if idx == 45:
                    #    self.action_space({'set_bus': {'substations_id': [(21, [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0])]}, 'redispatch': [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]})
                    #if idx == 56:    
                    #    self.action_space({'set_bus': {'substations_id': [(28, [1.0, 2.0, 1.0, 1.0, 2.0])]}, 'redispatch': [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]})
                    if idx == 45 or idx == 56:
                        if self.operationsequence[idx] == 1 and max(observation.rho) > 1:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(28, 2)]
                            action_space["set_bus"]["generators_id"] = [(9, 2)]
                            action_space["set_bus"]["loads_id"] = [(22, 2)]
                            action_space["redispatch"] = [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                            self.operationsequence[idx] += 1
                        if self.operationsequence[idx] == 0 and max(observation.rho) > 1:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(42, 2), (57, 2)]
                            action_space["set_bus"]["generators_id"] = [(16, 2)]
                            action_space["redispatch"] = [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                            self.operationsequence[idx] += 1
                    #if idx==39:
                    #    self.action_space({'set_bus': {'substations_id': [(21, [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0])]}, 'redispatch': [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]})
                    '''ESTA ACCION EMPEORA EN CODALAB
                    if idx == 39:
                        if self.operationsequence[idx] == 0 and max(observation.rho) > 1:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(31, 2)]
                            action_space["set_bus"]["lines_or_id"] = [(34, 2), (37, 2), (38, 2)]
                            action_space["set_bus"]["generators_id"] = [(11, 2)]
                            self.tooperateline = idx
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                            self.operationsequence[idx] += 1
                    '''
                    # 23 funciono muy bien en la plataforma
                    if idx == 23:
                        # if self.operationsequence[idx] == 1 and max(observation.rho) > 1:
                        #     action_space["set_bus"] = {}
                        #     action_space["set_bus"]["lines_ex_id"] = [(19, 2), (20, 2), (21, 2)]
                        #     action_space["set_bus"]["lines_or_id"] = [(22, 2), (48, 2), (49, 2)]
                        #     action_space["set_bus"]["loads_id"] = [(17, 2)]
                        #     action_space["set_bus"]["generators_id"] = [(7, 2)]
                        #     self.tooperateline = idx
                        #     print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                        #     self.operationsequence[idx] += 1
                        if self.operationsequence[idx] == 0 and max(observation.rho) > 1:

                            action_space_t1 = deepcopy(action_space)
                            target_lineidx, min_rho, target_action = self.line_search(observation, action_space_t1)
                            print("Curret Rho: ", max(observation.rho), "Disconnect Line #", target_lineidx, "for Line #", idx, 'Rho: ', min_rho)
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_or_id"] = [(34, 2), (37, 2)]
                            action_space["set_bus"]["generators_id"] = [(12, 2)]
                            action_space_t2 = deepcopy(action_space)
                            obs_, _, done, _ = observation.simulate(self.action_space(action_space_t2))
                            print("Curret Rho: ", max(observation.rho), "Adjust for Line #", idx, 'Rho: ', max(obs_.rho))
                            if max(obs_.rho) > min_rho:
                                action_space = deepcopy(target_action)
                                print(action_space)
                                self.operationoption[idx] = 1
                                self.operationlineaction[idx] = target_lineidx
                            else:
                                self.operationoption[idx] = 0
                            self.tooperateline = idx
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                            self.operationsequence[idx] += 1
                    if idx == 14:
                        # if self.operationsequence[idx] == 1 and max(observation.rho) > 1:
                        #     action_space["set_bus"] = {}
                        #     action_space["set_bus"]["lines_ex_id"] = [(2, 2)]
                        #     action_space["set_bus"]["lines_or_id"] = [(5, 2)]
                        #     self.tooperateline = idx
                        #     print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx])
                        #     self.operationsequence[idx] += 1
                        if self.operationsequence[idx] == 0 and max(observation.rho) > 1:
                            action_space_t1 = deepcopy(action_space)
                            target_lineidx, min_rho, target_action = self.line_search(observation, action_space_t1)
                            print("Curret Rho: ", max(observation.rho), "Disconnect Line #", target_lineidx, "for Line #", idx, 'Rho: ', min_rho)
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(19, 2), (21, 2)]
                            action_space["set_bus"]["lines_or_id"] = [(23, 2), (27, 2), (28, 2), (48, 2), (49, 2), (54, 2)]
                            action_space["set_bus"]["generators_id"] = [(5, 2), (6, 2), (7, 2), (8, 2)]
                            action_space_t2 = deepcopy(action_space)
                            obs_, _, done, _ = observation.simulate(self.action_space(action_space_t2))
                            print("Curret Rho: ", max(observation.rho), "Adjust for Line #", idx, 'Rho: ', max(obs_.rho))
                            if max(obs_.rho) > min_rho:
                                action_space = deepcopy(target_action)
                                print(action_space)
                                self.operationoption[idx] = 1
                                self.operationlineaction[idx] = target_lineidx
                            else:
                                self.operationoption[idx] = 0
                            self.tooperateline = idx
                            print("Adjust for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                            self.operationsequence[idx] += 1

            #for idx in line_disconnected[::-1]:
            #    if observation.time_before_cooldown_line[idx] == 0:
            #        lineidx = idx
            #        print("Reconnect #", lineidx)
            #        break

            for idx in line_disconnected[::-1]:
                if observation.time_before_cooldown_line[idx] == 0:
                    if not observation.line_status[23] and idx == self.operationlineaction[23]:
                        continue
                    if not observation.line_status[14] and idx == self.operationlineaction[14]:
                        continue
                    if not observation.line_status[27] and idx == self.operationlineaction[27]:
                        continue
                    lineidx = idx
                    break    

        for idx in range(self.nline):
            if observation.time_before_cooldown_line[idx] == 0:
                if idx == 45 or idx == 56:
                    if self.operationsequence[idx] == 1:
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(42, 1), (57, 1)]
                        action_space["set_bus"]["generators_id"] = [(16, 1)]
                        action_space["redispatch"] = [(10, -2.8), (13, -2.8), (19, 2.8), (21, 2.8)]
                        print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                        self.operationsequence[idx] -= 1
                    if self.operationsequence[idx] == 2:
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(28, 1)]
                        action_space["set_bus"]["generators_id"] = [(9, 1)]
                        action_space["set_bus"]["loads_id"] = [(22, 1)]
                        action_space["redispatch"] = [(10, -2.8), (13, -2.8), (19, 2.8), (21, 2.8)]
                        print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                        self.operationsequence[idx] -= 1
                #if idx==39:
                #    self.action_space({'set_bus': {'substations_id': [(21, [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0])]}, 'redispatch': [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]})
                '''ESTA ACCION EMPEORA EN CODALAB
                if idx == 39:
                    if self.operationsequence[idx] == 1:
                        action_space["set_bus"] = {}
                        action_space["set_bus"]["lines_ex_id"] = [(31, 1)]
                        action_space["set_bus"]["lines_or_id"] = [(34, 1), (37, 1), (38, 1)]
                        action_space["set_bus"]["generators_id"] = [(11, 1)]
                        self.tooperateline = idx
                        print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                        self.operationsequence[idx] -= 1
                '''
                # 23 funciono muy bien en la plataforma
                if idx == 23:
                    if self.operationsequence[idx] == 1:
                        if self.operationoption[idx] == 0:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_or_id"] = [(34, 1), (37, 1)]
                            action_space["set_bus"]["generators_id"] = [(12, 1)]
                            self.tooperateline = idx
                            self.operationsequence[idx] -= 1
                            print("Recover for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                        else:
                            if observation.line_status[idx]:
                                new_line_status_array = np.zeros(observation.rho.shape)
                                new_line_status_array[int(self.operationlineaction[idx])] = 1
                                action_space["set_line_status"] = new_line_status_array
                                self.tooperateline = idx
                                self.operationsequence[idx] -= 1
                                print("Recover for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                if idx == 14:
                    if self.operationsequence[idx] == 1:
                        if self.operationoption[idx] == 0:
                            action_space["set_bus"] = {}
                            action_space["set_bus"]["lines_ex_id"] = [(19, 1), (21, 1)]
                            action_space["set_bus"]["lines_or_id"] = [(23, 1), (27, 1), (28, 1), (48, 1), (49, 1), (54, 1)]
                            action_space["set_bus"]["generators_id"] = [(5, 1), (6, 1), (7, 1), (8, 1)]
                            self.tooperateline = idx
                            self.operationsequence[idx] -= 1
                            print("Recover for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])
                        else:
                            if observation.line_status[idx]:
                                new_line_status_array = np.zeros(observation.rho.shape)
                                new_line_status_array[int(self.operationlineaction[idx])] = 1
                                action_space["set_line_status"] = new_line_status_array
                                self.tooperateline = idx
                                self.operationsequence[idx] -= 1
                                print("Recover for Line #", idx, " Step: ", self.operationsequence[idx], "Option: #", self.operationoption[idx])

                    # if self.operationsequence[idx] == 2:
                    #     action_space["set_bus"] = {}
                    #     action_space["set_bus"]["lines_ex_id"] = [(2, 1)]
                    #     action_space["set_bus"]["lines_or_id"] = [(5, 1)]
                    #     self.tooperateline = idx
                    #     print("Recover for Line #", idx, " Step: ", self.operationsequence[idx])
                    #     self.operationsequence[idx] -= 1

        #if lineidx != -1:
        #    new_line_status_array[lineidx] = 1
        #    action_space["set_line_status"] = new_line_status_array
        #    obs_forecast, _, done, _ = observation.simulate(self.action_space(action_space))
        #    if not done and obs_forecast.rho.max() < observation.rho.max():
        #        pass
        #    else:
        #        return self.action_space({})

        if lineidx != -1:
            new_line_status_array = np.zeros(observation.rho.shape)
            new_line_status_array[lineidx] = 1
            tmpaction = {}
            tmpaction["set_line_status"] = new_line_status_array
            tmpres = self.action_space(tmpaction)
            obs_, _, done, _ = observation.simulate(tmpres)
            # if done or max(obs_.rho) > 1 > max(observation.rho):
            #     print("Can Not Reconnect #", lineidx, 'Estimate Rho:', max(obs_.rho))
            # else:
            action_space["set_line_status"] = new_line_status_array
            print("Reconnect #", lineidx, 'Estimate Rho:', max(obs_.rho))

        if self.tooperateline == 31:
            if self.operationsequence[31] == 1:
                self.operationsequence[31] -= 1
                self.operationtimestep[31] = 0
            else:
                self.operationsequence[31] += 1
                self.operationtimestep[31] = self.timestep
        
        res = self.action_space(action_space)
        assert res.is_ambiguous()

        
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
            action_space={}
            ltc_list = self.getRankedOverloads(observation)
            n_overloads = len(ltc_list)
            if n_overloads == 0:
                idx=0
            else:    
                idx=ltc_list[0]
            if idx < 21 or idx == 53 or idx == 54 or idx == 55:
                new_line_status_array = np.zeros(observation.rho.shape)
                new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                action_saved=self.action_space({'set_bus': {'substations_id': [(21, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]}, 'redispatch': [(2, -3.6), (3, -3.6), (13, 3.6), (16, 3.6)], 'set_line_status': new_line_status_array})
            else:
                new_line_status_array = np.zeros(observation.rho.shape)
                new_line_status_array[idx] = 0 #revisar como se desconecta la linea
                action_saved=self.action_space({'set_bus': {'substations_id': [(21, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]}, 'redispatch': [(2, 3.6), (3, 3.6), (13, -3.6), (16, -3.6)], 'set_line_status': new_line_status_array})
            #res=action_saved
            best_action=action_saved

            
        print("action we take is:")
        print(res)
        #return res
        return best_action 

    #from amarot
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

    #from Magic Power Grids
    def line_search(self, observation, action_space):
        new_line_status_array = np.zeros(observation.rho.shape)
        min_rho = max(observation.rho)
        target_lineidx = -1
        target_action = self.action_space(action_space)
        for lineidx in range(self.nline):
            new_line_status_array[lineidx] = -1
            action_space["set_line_status"] = new_line_status_array
            res = self.action_space(action_space)
            obs_, _, done, _ = observation.simulate(res)
            if not done and max(obs_.rho) < min_rho:
                target_lineidx = lineidx
                min_rho = max(obs_.rho)
                target_action = deepcopy(action_space)
            new_line_status_array[lineidx] = 0
        return target_lineidx, min_rho, target_action              

def make_agent(env, submission_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your sudmission directory and return a valid agent.
    """
    res = ReconnectAgent(env.action_space, env.observation_space)
    return res
