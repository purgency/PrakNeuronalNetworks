import gym
import numpy
numpy.random.seed()
env = gym.make('Breakout-ram-v0')
print(env.action_space)
print(env.observation_space)
env.reset()

wfeat = numpy.random.uniform(0.0, 1.0, (20,128)) -0.5
wact = numpy.random.uniform(0.0, 1.0, (6,20)) -0.5
wrec = numpy.random.uniform(0.0, 1.0, (20,20)) -0.5

bfeat = -0.5* numpy.random.uniform(0.0, 1.0, 20)
bact = 0.0* numpy.random.uniform(0.0, 1.0, 6)
learningrate = 0.03

accum_rewards = 0.0

transferfunc = lambda h: numpy.tanh(h) #tanh
ableitungtransfunc = lambda h: 1 - numpy.power(transferfunc(h),2)
#transferfunc = lambda h: 1.0/(1.0+numpy.exp(-h)) #sigmoid
#ableitungtransfunc = lambda h: transferfunc(h) * (1.0-transferfunc(h))
#transferfunc = lambda h: numpy.log(1+numpy.exp(h)) #relu
#ableitungtransfunc = lambda h: 1/(1 + numpy.exp(-h))

for i_episode in range(100000):
    if i_episode % 3 == 0:
        inputs = []
        shids = []
        h1s = []
        counter = 0
    observation = env.reset()

    randvar = 1.0 / (0.3*i_episode+1)

    old_Input = observation

    current_h1 = numpy.dot(wfeat,old_Input)+bfeat
    current_shid = transferfunc(current_h1)
    current_h2 = numpy.dot(wact,current_shid)+bact
    current_sout = current_h2
    
    inputs.insert(0,old_Input)
    shids.insert(0,current_shid)
    h1s.insert(0,current_h1)

    if(numpy.random.rand() <= randvar):
        current_action = env.action_space.sample()
    else:
        current_action = numpy.argmax(current_sout)

    current_Q = current_sout[current_action]
    for t in range(10000):
        if i_episode >= 0:
            env.render()
        new_observation, reward, done, info = env.step(current_action)
        if done and reward == 0.0:
            reward = -0.2
        accum_rewards += reward

        #print new_observation
        #raw_input("Press Enter to continue...")

        Input = new_observation

        new_h1 = numpy.dot(wrec,current_shid)+numpy.dot(wfeat,Input)+bfeat
        new_shid = transferfunc(new_h1)
        new_h2 = numpy.dot(wact,new_shid)+bact
        new_sout = new_h2
        
        inputs.insert(0,Input)
        shids.insert(0,new_shid)

        if(numpy.random.uniform() <= randvar):
            next_action = env.action_space.sample()
        else:
            next_action = numpy.argmax(new_sout)

        new_Q = new_sout[next_action]
        
        delta1s = []

        delta2 = numpy.zeros(6)
        delta2[current_action] = reward + 0.9*new_Q - current_Q
        delta1 = ableitungtransfunc(current_h1) * numpy.dot(delta2, wact)
        delta1s.insert(0,delta1)
        h1s.insert(0,new_h1)
        for x in range(counter):
            if True:
                delta1s.insert(0,ableitungtransfunc(h1s[x+1])*numpy.dot(delta1s[0],wrec))
        wact  += learningrate * numpy.outer(delta2, current_shid)
        for x in range(counter):
            if True:
                wfeat += learningrate * numpy.outer(delta1s[-x], inputs[x+1])
        for x in range(counter):
            if True:
                wrec += learningrate * numpy.outer(delta1s[-x], shids[x+1])
        bact  += learningrate * delta2
        for x in range(counter):
            if True:
                bfeat += learningrate * delta1s[-x]

        current_action = next_action
        current_Q = new_Q
        current_h1 = new_h1
        current_shid = new_shid
        old_Input = Input
        counter += 1

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1), accum_rewards)
            break