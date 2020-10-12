#!/usr/bin/env python3
# encoding utf-8

import sys
sys.path.insert(0,'/afs/inf.ed.ac.uk/user/s18/s1877727/HFO/')
sys.path.insert(0,'/home/timos/HFO')


# sys.path.insert(0, '/home/timos/HFO')

from hfo import *
import random
import argparse
from HFOGoalkeepingPlayer import HFOGoalkeepingPlayer


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--port', type=int, default=6000)
	parser.add_argument('--numEpisodes', type=int, default=1000)
	args=parser.parse_args()

	numEpisodes = args.numEpisodes

	hfo = HFOGoalkeepingPlayer()
	hfo.connectToServer()

	for episode in range(numEpisodes):	
		status = 0
		while status==0:
			status = hfo.moveToCorner()

		if status == 5:
			hfo.hfo.act(QUIT)
			break


