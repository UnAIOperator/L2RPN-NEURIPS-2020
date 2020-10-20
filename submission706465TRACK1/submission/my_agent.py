'''
TEAM: UN_aiGridOperator

TEAM MEMBERS

Daniel Alejandro Gualteros Gualteros <dgualterosg@unal.edu.co>, student at Universidad Nacional de Colombia
Edgard Leonardo Castaneda Garcia <elcastanedag@unal.edu.co>, student at Universidad Nacional de Colombia
David Leonardo Alvarez Alvarez <dlalvareza@unal.edu.co>, Associate Postdoc at Universidad Nacional de Colombia
Ivan Felipe Bonilla Vargas, Enel Colombia<ivan.bonilla@enel.com>, Senior Engineer at Enel-Codensa
Sergio Raul Rivera Rodriguez <srriverar@unal.edu.co>, Associate Professor at Universidad Nacional de Colombia 

first approach some functions from amarot (the simulation time is improved) plus expert knowledge actions from heuristic action from simulations
'''
import numpy as np
from grid2op.Agent import BaseAgent
import pandapower as pp

class ReconnectAgent(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """
    def __init__(self, action_space, observation_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        
        #from amorot
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

        self.action_space = action_space
        
    def act(self, observation, reward, done):
        """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
        self.curr_iter += 1

        #best_action based on amarot some functions, here it is improved the whole simulation time
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

        #action_space base in expert knowledge (based on heuristic actions on local simulations)
        action_space = {}

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
                    if idx == 45:
                        self.action_space({'set_bus': {'substations_id': [(21, [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0])]}, 'redispatch': [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]})
                    if idx == 56:    
                        self.action_space({'set_bus': {'substations_id': [(28, [1.0, 2.0, 1.0, 1.0, 2.0])]}, 'redispatch': [(10, 2.8), (13, 2.8), (19, -2.8), (21, -2.8)]})

            for idx in line_disconnected[::-1]:
                if observation.time_before_cooldown_line[idx] == 0:
                    lineidx = idx
                    break

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
        return res 

    #FUNCTIONS BASED ON AMAROT
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
