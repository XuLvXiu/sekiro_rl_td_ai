#encoding=utf8

class Rule(): 
    '''
    predefined state rule, these state no need to be explored.
    '''

    def apply(self, state, env): 
        '''
        apply rule, get action id(not game action id)
        '''
        # 10
        base_offset = env.action_space + env.RULE_COUNT

        # predefined state
        # state-6: 
        if state.state_id == env.state_manager.HULU_STATE_ID: 
            # 9
            action_id = base_offset - 1
            return action_id

        # state-5: 
        if state.state_id == env.state_manager.PLAYER_HP_DOWN_STATE_ID: 
            # 8
            action_id = base_offset - 2
            return action_id

        # classification model
        # state-0: class-0
        if state.state_id == env.state_manager.NORMAL_STATE_ID: 
            # 7
            action_id = base_offset - 3
            return action_id

        # state-4: class-4
        if state.state_id == env.state_manager.BAD_TUCI_STATE_ID: 
            # 7
            action_id = base_offset - 3
            return action_id

        # no rule found
        return None


