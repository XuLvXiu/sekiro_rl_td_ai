#encoding=utf8

'''
collect image data manually, but not collect actions.
used for classification model
'''
print('importing...')


import sys
from log import log
import cv2
import time
import signal
from pynput.keyboard import Listener, Key
import numpy as np
import os
import time
import os
import shutil
import argparse
from env import Env


class DataCollector: 
    '''
    collect data for the cluster model
    '''

    def __init__(self, is_resume=True): 
        '''
        init
        '''
        log.info('is_resume: %s' % (is_resume))
        image_dir = 'images/original'
        if not is_resume: 
            log.info('delete old images')
            if os.path.exists(image_dir): 
                shutil.rmtree(image_dir)

        if not os.path.exists(image_dir): 
            os.mkdir(image_dir)

        self.env = Env()
        self.env.eval()


    def run(self): 
        '''
        main process
        '''
        time_column = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        episode = self.generate_episode()

        log.info('flush to disk, pls do NOT power off...')
        for i in range(0, len(episode)): 
            state = episode[i][0]
            image = state.image
            cv2.imwrite('images/original/%s_%s.png' % (time_column, i), 
                image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        log.info('flush done')


    def generate_episode(self): 
        '''
        generate an episode 
        '''
        global g_episode_is_running

        episode = []
        env = self.env

        env.reset()
        state = env.get_state()

        g_episode_is_running = False

        step_i = 0

        while True: 
            # log.info('generate_episode main loop running')
            if not g_episode_is_running: 
                print('press ] to begin the collector, press again to flush images and exit')
                time.sleep(1.0)
                env.reset()
                state = env.get_state()
                continue

            t1 = time.time()
            log.info('generate_episode step_i: %s,' % (step_i))

            state = env.get_state()
            action_id = None
            reward = None

            t2 = time.time()
            log.info('generate_episode main loop end one epoch, time: %.2f s' % (t2-t1))
            step_i += 1

            # save to current episode
            episode.append((state, action_id, reward))

            if not g_episode_is_running: 
                log.info('done.')
                break

        # end of while loop

        log.info('episode done. length: %s' % (len(episode)))
        return episode


    def stop(self): 
        '''
        stop the collector
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
                g_episode_is_running = False
                t.stop()
            else: 
                g_episode_is_running = True

    except Exception as e:
        print(e)

signal.signal(signal.SIGINT, signal_handler)
keyboard_listener = Listener(on_press=on_press)
keyboard_listener.start()

t = DataCollector(is_resume)
t.run()
