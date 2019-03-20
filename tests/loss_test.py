import sys
import os
sys.path.append(os.path.abspath(os.path.join('../', 'src')))
import pytest


def test_calc_loss_dqn():

	assert (1+1) == 2