#encoding=utf8

print('importing...')


from env import Env
import sys
import torch
from log import log
import cv2
import time
import signal
from pynput.keyboard import Listener, Key
import numpy as np
import os
import pickle
import json
import time
import argparse
from storage import Storage
from rule import Rule

class Trainer: 
    '''
    train a Temporal-Difference(TD) agent.
    '''

    def __init__(self, is_resume=True): 
        '''
        init
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info('device: %s' % (self.device))

        self.env = Env()
        self.env.train()

        # number of actions to be explored.
        self.action_space = self.env.action_space

        # TD paramaters
        # the length of Q[s] and N[s] 
        # should be number of actions to be explored + number of actions not to be explored
        d2_length = self.env.action_space + self.env.RULE_COUNT
        self.Q = Storage(d2_length)
        self.GAMMA = 0.85
        self.ALPHA = 0.5

        # extra paramaters for debug
        self.N = Storage(d2_length)

        # episode parameters
        self.MAX_EPISODES = 1000
        self.next_episode = 0
        self.CHECKPOINT_FILE = 'checkpoint.pkl'
        self.JSON_FILE = 'checkpoint.json'

        log.info('is_resume: %s' % (is_resume))
        if is_resume: 
            obj_information = self.load_checkpoint()
            self.next_episode = obj_information['episode']

        self.env.create_game_status_window()


    def train(self): 
        '''
        mc control
        '''
        begin_i = self.next_episode
        for i in range(begin_i, self.MAX_EPISODES): 
            # decay
            epsilon = 1.0 / ((i+1) / self.MAX_EPISODES + 1)
            log.info('episode: %s, epsilon: %s' % (i, epsilon))
            self.env.game_status.episode = i

            # one episode
            episode = self.generate_episode_from_Q_and_update_Q(epsilon)

            self.next_episode += 1

            obj_information = {
                'episode': self.next_episode,
                'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            }
            self.save_checkpoint(obj_information)

            self.env.go_to_next_episode()

        log.info('mission accomplished :)')
    

    def select_action_using_Q(self, state, epsilon): 
        '''
        select an action by state using Q
        '''
        if self.Q.has(state): 
            log.info('select_action_using_Q: state[%s] found, using epsilon-greedy' % (str(self.Q.convert_state_to_key(state))))
            Q_s = self.Q.get(state)
            probs = self.get_probs(Q_s, epsilon)
            log.info('Q_s: %s' % (Q_s))
            log.info('probs: %s' % (probs))
            action_id = np.random.choice(self.action_space, p=probs)
            return action_id


        # if Q does not have state, use random
        log.info('select_action_using_Q: state[%s] not found, using random' % (str(self.Q.convert_state_to_key(state))))
        action_id = np.random.randint(0, self.action_space)
        return action_id


    def select_action(self, state, epsilon): 
        '''
        select an action by state using Q, model and rules.
        '''
        # rules: 
        # state-5
        # state-6
        # state-0: class-0
        # state-4: class-4
        obj_rule = Rule()
        action_id = obj_rule.apply(state, self.env)
        if action_id is not None: 
            log.info('select_action: state[%s], using predefined rule: [%s]' % (str(self.Q.convert_state_to_key(state)), action_id))
            return action_id

        # Q: 
        # state-1/2/3: class-1/2/3
        action_id = self.select_action_using_Q(state, epsilon)
        return action_id


    def generate_episode_from_Q_and_update_Q(self, epsilon): 
        '''
        generate an episode using epsilon-greedy policy
        and update Q after each step
        '''
        env = self.env

        # init S
        env.reset()
        state = env.get_state()

        step_i = 0

        # select action(A) by state(S) using Q
        action_id = self.select_action(state, epsilon)
        next_action_id = None

        while True: 
            # log.info('generate_episode main loop running')
            if not g_episode_is_running: 
                print('if you lock the boss already, press ] to begin the episode')
                self.env.game_status.is_ai = False
                self.env.update_game_status_window()
                time.sleep(1.0)

                env.reset()
                state = env.get_state()

                step_i = 0

                # select action(A) by state(S) using Q
                action_id = self.select_action(state, epsilon)
                next_action_id = None
                continue

            self.env.game_status.is_ai = True

            t1 = time.time()
            log.info('generate_episode step_i: %s,' % (step_i))

            self.env.game_status.step_i     = step_i
            self.env.game_status.error      = ''
            self.env.game_status.state_id   = self.Q.convert_state_to_key(state)
            self.env.update_game_status_window()

            # take action(A), get reward(R) and next state(S')
            # at first, convert rf action_id to game action_id
            game_action_id = self.env.arr_possible_action_id[action_id]
            next_state, reward, is_done = env.step(game_action_id)

            # get next action(A') by next_state(S') using Q
            next_action_id = self.select_action(next_state, epsilon)

            # sarsa = (S, A, R, S', A')
            sarsa = (state, action_id, reward, next_state, next_action_id)
            self.update_Q(sarsa, is_done)

            # prepare for next step
            # S = S'
            # A = A'
            state = next_state
            action_id = next_action_id

            t2 = time.time()
            log.info('generate_episode main loop end one step, time: %.2f s' % (t2-t1))
            step_i += 1

            if is_done: 
                env.stop()
                log.info('done.')
                break

        # end of while loop

        log.info('episode done. length: %s' % (step_i))
        self.env.update_game_status_window()


    def get_probs(self, Q_s, epsilon): 
        '''
        obtain the action probabilities related to the epsilon-greedy policy.
        '''
        ones = np.ones(self.action_space)
        # default action probability
        policy_s = ones * epsilon / self.action_space

        # best action probability
        a_star = np.argmax(Q_s)
        log.info('a_star: %s' % (a_star))
        policy_s[a_star] = 1 - epsilon + epsilon / self.action_space

        return policy_s


    def update_Q(self, sarsa, is_done): 
        '''
        update Q using sarsa
        '''
        (state, action_id, reward, next_state, next_action_id) = sarsa

        Q_s = self.Q.get(state).copy()
        Q_s_next = self.Q.get(next_state)

        Q_s_a = Q_s[action_id]
        Q_s_a_next = Q_s_next[next_action_id]

        # if S t+1 is the end of the episode, then Q(S t+1, A t+1) = 0
        if is_done: 
            Q_s_a_next = 0

        # Q(S, A) = Q(S, A) + alpha * (R + gamma * Q(S', A') - Q(S, A))
        new_value = Q_s_a + self.ALPHA * (reward + self.GAMMA * Q_s_a_next - Q_s_a)
        self.Q.set(state, action_id, new_value)

        # for debug
        N_s = self.N.get(state).copy()
        old_cnt = N_s[action_id]
        new_cnt = old_cnt + 1
        self.N.set(state, action_id, new_cnt)

        log.debug('''update_Q: 
                old_Q_s[%s] old_N_s[%s], Q_s_next[%s],
                state[%s] action[%s] reward[%s] next_state[%s] next_action[%s] next_Q_s_a[%s] 
                new_Q_s[%s] new_N_s[%s]''' % (Q_s, N_s, Q_s_next,
            str(self.Q.convert_state_to_key(state)), action_id, reward,
            str(self.Q.convert_state_to_key(next_state)), next_action_id, Q_s_a_next,
            self.Q.get(state), self.N.get(state)))


    def save_checkpoint(self, obj_information): 
        '''
        save checkpoint for future use.
        '''
        log.info('save_checkpoint...')
        log.info('Q: %s' % (self.Q.summary('Q')))
        log.info('N: %s' % (self.N.summary('N')))
        log.info('do NOT terminate the power, still saving...')
        # log.info('actions: %s' % (self.env.arr_action_name))
        
        # pickle Q and N
        with open(self.CHECKPOINT_FILE, 'wb') as f:
            pickle.dump((self.Q, self.N), f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info('still saving...')

        # write json information
        log.debug(obj_information)
        with open(self.JSON_FILE, 'w', encoding='utf-8') as f: 
            json.dump(obj_information, f, indent=4, ensure_ascii=False) 

        log.info('saved ok')


    def load_checkpoint(self): 
        '''
        load history checkpoint
        '''
        log.info('load_checkpoint')
        obj_information = {'episode': 0}
        try: 
            with open(self.CHECKPOINT_FILE, 'rb') as f: 
                (self.Q, self.N) = pickle.load(f)

            with open(self.JSON_FILE, 'r', encoding='utf-8') as f: 
                obj_information = json.load(f)
        except Exception as e: 
            log.error('ERROR load checkpoint: %s', (e))

        log.info('Q: %s' % (self.Q.summary('Q')))
        log.info('N: %s' % (self.N.summary('N')))
        log.debug(obj_information)
        return obj_information


    def stop(self): 
        '''
        stop the trainer
        '''
        self.env.stop()



# main
parser = argparse.ArgumentParser()
parser.add_argument('--new', action='store_true', help='new training', default=False)
args = parser.parse_args()
is_resume = not args.new

g_episode_is_running = False
def signal_handler(sig, frame):
    log.debug("Gracefully exiting...")
    t.stop()
    sys.exit(0)

def on_press(key):
    # print('on_press: %s' % (key))
    global g_episode_is_running
    try:
        if key == Key.backspace: 
            log.info('The user presses backspace in the game, will terminate.')
            t.stop()
            os._exit(0)

        if hasattr(key, 'char') and key.char == ']': 
            # switch the switch
            if g_episode_is_running: 
                # g_episode_is_running = False
                # t.stop()
                print('I cannot stop myself lalala')
            else: 
                g_episode_is_running = True

    except Exception as e:
        print(e)

signal.signal(signal.SIGINT, signal_handler)
keyboard_listener = Listener(on_press=on_press)
keyboard_listener.start()

t = Trainer(is_resume)
t.train()
