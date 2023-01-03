# This file contains the options that you should modify to solve Question 2

"""
the bigger the living reward, the more risky the model is
"""
def question2_1():
    """
    we seek the near terminal state (reward + 1)
    via the short dangerous path (exit quickly)
    """
    return {
        "noise": 0.001,
        "discount_factor": 0.1,
        "living_reward": -1
    }


def question2_2():
    """
    we seek the near terminal state (reward + 1)
    via the long safe path
    """
    return {
        "noise": 0.4,
        "discount_factor": 0.5,
        "living_reward": -1
    }


def question2_3():
    """
    we seek the near terminal state (reward + 10)
    via the short dangerous path
    """
    return {
        "noise": 0.1,
        "discount_factor": 0.9,
        "living_reward": -1
    }


def question2_4():
    """
    we to seek the near terminal state (reward + 10)
    via the long safe path
    """
    return {
        "noise": 0.1,
        "discount_factor": 1,
        "living_reward": -0.1
    }


def question2_5():
    """
    we want to avoid any terminal state and keep the episode going forever
    """
    return {
        "noise": 0.1,
        "discount_factor": 0.9,
        "living_reward": 2
    }


def question2_6():
    """
    we seek any terminal state
    """
    return {
        "noise": 0.2,
        "discount_factor": 0.9,
        "living_reward": -15
    }
