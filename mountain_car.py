import gym
from gym import wrappers
from tile3 import IHT,tiles
import random
import numpy as np

maxSize = 5000
iht = IHT(maxSize)
weights = [0]*maxSize
numTilings = 8
learningRate = 0.0125

def mytiles(position, velocity,action=[]):
    scale_P = 5/1.7
    scale_V =  5/.14
    return tiles(iht, numTilings, list((position*scale_P,velocity*scale_V)),action)

def Q_estimate(position, velocity, action=[]):
    tiles = mytiles(position, velocity,action)
    estimate = 0
    for tile in tiles: 
        estimate += weights[tile]
    return estimate 

def Q_learn(position, velocity,q_td_target, action=[]):
    tiles = mytiles(position, velocity,action)
    estimate = 0
    for tile in tiles: 
        estimate += weights[tile]                  #form estimate
    error = q_td_target - estimate
    for tile in tiles: 
        weights[tile] += learningRate * error          #learn weights


def getMaxIndex(qValues):
    return np.argmax(qValues)

# select the action with the highest Q value
def selectAction(qValues, explorationRate):
    rand = random.random()
    if rand < explorationRate :
            action = np.random.randint(0, 3)
    else :
            action = getMaxIndex(qValues)
    return action


env = gym.make('MountainCar-v0')
outdir = '/tmp/mountaincar-results'
env = wrappers.Monitor(env, outdir, force=True,video_callable=lambda episode_id: episode_id%10==0)

epochs = 100
steps = 220
updateTargetNetwork = 10000
explorationRate = 0.0
stepCounter =0
discountFactor = 0.99

qValues=np.zeros(3,dtype=float)

# Main loop
for epoch in range(epochs):
    observation = env.reset()
    print(explorationRate)
    # number of timesteps
    done= False
    env.render()
    qValues[0] = Q_estimate(observation[0],observation[1],[0])
    qValues[1] = Q_estimate(observation[0],observation[1],[1])
    qValues[2] = Q_estimate(observation[0],observation[1],[2])
    action = selectAction(qValues, explorationRate)
    # Episode loop (inner loop)
    for t in range(steps):

        newObservation, reward, done, info = env.step(action)
        env.render()
        if (done):
            target=reward
        else:
            qValues[0] = Q_estimate(newObservation[0],newObservation[1],[0])
            qValues[1] = Q_estimate(newObservation[0],newObservation[1],[1])
            qValues[2] = Q_estimate(newObservation[0],newObservation[1],[2])
            new_action = selectAction(qValues, explorationRate)
            target = reward + discountFactor * Q_estimate(newObservation[0],
                                                          newObservation[1],
                                                          [new_action])
        Q_learn(observation[0],observation[1],target,[action])

        observation = newObservation
        action = new_action
        stepCounter += 1
        if (done) : break
    explorationRate *= 0.995
    explorationRate = max (0.0, explorationRate)

    print("Steps so far=",stepCounter)
