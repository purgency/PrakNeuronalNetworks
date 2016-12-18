import gym
import numpy
import KTimage as kt

numpy.random.seed()

env = gym.make('FrozenLake-v0')

#SFFF 0123 left 0
#FHFH 4567 down 1
#FFFH 89AB right 2
#HFFG CDEF up 3

print(env.action_space)

env.__init__(is_slippery=False)
env.render()

wfeat = numpy.random.uniform(0.0, 1.0, (10,16)) -0.5
wact = numpy.random.uniform(0.0, 1.0, (4,10)) -0.5

bfeat = -0.5* numpy.random.uniform(0.0, 1.0, 10)
bact = 0.0* numpy.random.uniform(0.0, 1.0, 4)
learningrate = 0.3

accum_rewards = 0.0

transferfunc = lambda h: 1.0/(1.0+numpy.exp(-h))
ableitungtransfunc = lambda h: transferfunc(h) * (1.0-transferfunc(h))

episodes = 50000

for i_episode in range(episodes):
    observation = env.reset()

    randvar = 0.1

    old_Input = numpy.zeros(16)
    old_Input[observation] = 1.0

    current_h1 = numpy.dot(wfeat,old_Input)+bfeat
    current_shid = transferfunc(current_h1)
    current_h2 = numpy.dot(wact,current_shid)+bact
    current_sout = current_h2

    if(numpy.random.rand() <= randvar):
        current_action = env.action_space.sample()
    else:
        current_action = numpy.argmax(current_sout)

    current_Q = current_sout[current_action]

    for t in range(100):
        new_observation, reward, done, info = env.step(current_action)
        if  done and (reward == 0.0):
                reward = -0.5
        accum_rewards += reward

        Input = numpy.zeros(16)
        Input[new_observation] = 1.0

        new_h1 = numpy.dot(wfeat,Input)+bfeat
        new_shid = transferfunc(new_h1)
        new_h2 = numpy.dot(wact,new_shid)+bact
        new_sout = new_h2

        if(numpy.random.uniform() <= randvar):
                next_action = env.action_space.sample()
        else:
                next_action = numpy.argmax(new_sout)

        new_Q = new_sout[next_action]

        delta2 = numpy.zeros(4)
        delta2[current_action] = reward + 0.9*new_Q - current_Q
        delta1 = ableitungtransfunc(current_h1) * numpy.dot(delta2, wact)
        wact  += learningrate * numpy.outer(delta2, current_shid)
        wfeat += learningrate * numpy.outer(delta1, old_Input)
        bact  += learningrate * delta2
        bfeat += learningrate * delta1

        current_action = next_action
        current_Q = new_Q
        current_h1 = new_h1
        current_shid = new_shid
        old_Input = Input

        if done and (i_episode % 100 == 0 or i_episode > episodes - 10):
                print("Episode {} finished after {} timesteps".format(i_episode, t+1), accum_rewards)
                kt.exporttiles(array=new_sout, height=1, width=4, filename="results/obs_S_2.pgm")
                kt.exporttiles(array=wact, height=1, width=10, outer_height=1, outer_width=4, filename="results/obs_W_2_1.pgm")
                kt.exporttiles(array=wfeat, height=4, width=4, outer_height=1, outer_width=10, filename="results/obs_W_1_0.pgm")
                kt.exporttiles(array=numpy.dot(wact,wfeat), height=4, width=4, outer_height=1, outer_width=4, filename="results/obs_W_2_0.pgm")
                env.render()

        if done:
                break