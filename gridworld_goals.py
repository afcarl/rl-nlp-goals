import numpy as np
import random
import itertools
import scipy.ndimage
import scipy.misc
import matplotlib.pyplot as plt


class gameOb():
    def __init__(self, coordinates, size, color, reward, name):
        '''
        Generate an object to add at the envirments
        :param coordinates: initial position of the obj
        :param size: size of the object
        :param color: indication of the measurements vector
        :param reward: reward given by the obj
        :param name: name of the obj
        '''
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.color = color
        self.reward = reward
        self.name = name

class gameEnv():
    def __init__(self, partial, env_size, a_size):
        '''
        Set the games enviroments
        :param partial: if using or not using the padding
        :param env_size: size of the envoriments (matrix x-by-x)
        :param a_size: action size
        '''
        self.sizeX = env_size
        self.sizeY = env_size
        self.actions_size = a_size
        assert self.actions_size <= 4
        self.objects = []
        self.partial = partial
        self.bg = np.zeros([env_size, env_size])                    # battle-ground used for the game
        a, a_big, measurements, goal, hero = self.reset()


    def getFeatures(self):
        '''
        :return: get the position of the landing
        '''
        return np.array([self.objects[0].x,self.objects[0].y]) / float(self.sizeX)

    def reset(self):
        '''
        reset the enviroments, adding a drone an a goal
        :return:  return the enviroment, the enviroments images, the measurements value, the goal position and the drone position
        '''
        self.objects = []
        self.orientation = 0
        self.hero = gameOb(self.newPosition(), 1, [1, 1, 1], None, 'hero')     # added the dron

        self.measurements = [0.0, 1.0]                                          # reset the measurements (expression of the goal)

        self.objects.append(self.hero)
        battery = gameOb([0, 0], 1, [0, 0, 1], 1, 'battery')                    # generate a battery position
        self.battery = battery
        self.objects.append(battery)
        for i in range(1):
            bug = gameOb(self.newPosition(), 1, [0, 1, 0], 1, 'goal')          # generate a delivery position
            self.objects.append(bug)
        self.goal = bug
        state, s_big = self.renderEnv()
        self.state = state
        return state, s_big, self.measurements, [self.goal.x, self.goal.y], [self.hero.x, self.hero.y]

    def moveChar(self, action):
        '''
        there are 6 actions:
        - 0 - up
        - 1 - down
        - 2 - left
        - 3 - right
        - 4 - 90 counter-clockwise
        - 5 - 90 clockwise
        :param action: action to take
        :return: new position of the drone according to the action done, the penality encountered during the action taken
        '''
        hero = self.objects[0]
        block_positions = [[-1, -1]]  # take in consideration all the blocks along the env
        for ob in self.objects:
            if ob.name == 'block': block_positions.append([ob.x, ob.y])
        blockPositions = np.array(block_positions)
        hero_x = hero.x
        hero_y = hero.y
        penalize = 0.
        if action < self.actions_size:
            if self.orientation == 0:
                direction = action
            if self.orientation == 1:
                if action == 0: direction = 1
                elif action == 1: direction = 0
                elif action == 2: direction = 3
                elif action == 3: direction = 2
            if self.orientation == 2:
                if action == 0: direction = 3
                elif action == 1: direction = 2
                elif action == 2: direction = 0
                elif action == 3: direction = 1
            if self.orientation == 3:
                if action == 0: direction = 2
                elif action == 1: direction = 3
                elif action == 2: direction = 1
                elif action == 3: direction = 0

            if direction == 0 and hero.y >= 1 and [hero.x, hero.y - 1] not in blockPositions.tolist():
                hero.y -= 1
            if direction == 1 and hero.y <= self.sizeY-2 and [hero.x, hero.y + 1] not in blockPositions.tolist():
                hero.y += 1
            if direction == 2 and hero.x >= 1 and [hero.x - 1,hero.y] not in blockPositions.tolist():
                hero.x -= 1
            if direction == 3 and hero.x <= self.sizeX-2 and [hero.x + 1,hero.y] not in blockPositions.tolist():
                hero.x += 1

        if hero.x == hero_x and hero.y == hero_y:   # not move
            penalize = 0.0
        self.objects[0] = hero
        self.hero = hero
        return penalize

    def newPosition(self):
        '''
        get a new empty position
        '''
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):             # generate all the possible position
            points.append(t)
        for obj in self.objects:                            # remove the occupied one
            if (obj.x, obj.y) in points:
                points.remove((obj.x, obj.y))
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def checkGoal(self):
        '''
        check if we get some rewards
        :return: reward obtained, a boolean
        '''
        if len(self.objects) > 1:
            hero = self.hero
            others = self.objects[1:]
            ended = False
            for other in others:
                if hero.x == other.x and hero.y == other.y and hero != other:
                    self.objects.remove(other)
                    ended == True
                    if other.name == 'goal':
                        goal = gameOb(self.newPosition(), 1, [0, 1, 0], 1, 'goal')
                        self.objects.append(goal)
                        self.goal = goal
                        self.measurements[0] += 1
                        return other.reward, False
                    if other.name == 'battery':
                        battery = gameOb([0, 0], 1, [0, 0, 1], 1, 'battery')
                        self.objects.append(battery)
                        self.battery = battery
                        self.measurements[1] = 1.0
                        return other.reward, False
            if ended == False:
                return 0.0, False
        else:
            return 0.0, False

    def renderEnv(self):
        '''
        reset the enviroments, generating the enviroment and the enviroments image with all the icons.
        padding is used to make sure that the icon fits the enviroments
        :return: env, env image
        '''
        if self.partial == True:    # check if we use the padding
            padding = 2
            a = np.ones([self.sizeY+(padding*2), self.sizeX+(padding*2), 3])
            a[padding:-padding, padding:-padding, :] = 0
            a[padding:-padding, padding:-padding, :] += np.dstack([self.bg, self.bg, self.bg])
        else:
            a = np.zeros([self.sizeY, self.sizeX, 3])
            padding = 0
            a += np.dstack([self.bg, self.bg, self.bg])
        hero = self.objects[0]
        for item in self.objects:
            a[item.y+padding: item.y+item.size+padding, item.x+padding: item.x+item.size+padding, :] = item.color
        if self.partial == True:
            a = a[hero.y: (hero.y+(padding*2)+hero.size), hero.x: (hero.x+(padding*2)+hero.size), :]
        a_big = a
        a_big[self.goal.y+padding: self.goal.y+self.goal.size+padding,
        self.goal.x+padding: self.goal.x+self.goal.size+padding, :] = [0, 1, 0]
        a_big = scipy.misc.imresize(a_big,[32,32,3],interp='nearest')
        return a, a_big

    def step(self, action):
        '''
        execute one step
        :param action: action to execute
        :return: env, env_image, reward obtained, position of the goal, position of the hero
        '''
        if self.objects != []:
            penalty = self.moveChar(action)     # move the hero
        reward, done = self.checkGoal()
        self.measurements[1] -= 0.025           # decrease the battery
        if self.measurements[1] <= 0:
            done = True                         # got the delivery
            self.measurements[1] = 0.0
        state, s_big = self.renderEnv()         # render the enviroments
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state, self.measurements, done
        else:
            goal = None
            for ob in self.objects:
                if ob.name == 'goal':
                    goal = ob
            return state, s_big, self.measurements, [self.goal.x, self.goal.y], [self.hero.x, self.hero.y], done
