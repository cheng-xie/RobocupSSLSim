import sys, math, random
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding

FPS    = 50
SCALE  = 10.0   # affects how fast-paced the game is, forces should be adjusted as well

INITIAL_RANDOM = 1.0   # Set 1500 to make game harder

VIEWPORT_W = 800
VIEWPORT_H = 600

'''
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.lander==contact.fixtureA.body or self.env.lander==contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False
'''

class SSLSimpleNav(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self):
        self._seed()
        self.viewer = None

        self.world = Box2D.b2World((0,0))
        self.moon = None
        self.robots = [] 

        self.prev_reward = None

        high = np.array([np.inf]*8)  # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-high, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Up-Down: -1.0..-0.5 fire down engine, +0.5..+1.0 fire up engine, -0.5..0.5 off
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,))
        else:
            # Nop, fire left engine, up engine, right engin, down
            self.action_space = spaces.Discrete(5)

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        #self.world.contactListener = None
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.robot)
        self.robot = None
        #self.world.DestroyBody(self.legs[0])
        #self.world.DestroyBody(self.legs[1])

    def _reset(self):
        self._destroy()
        #self.world.contactListener_keepref = ContactDetector(self)
        #self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE
        
        self.drawlist = []
        
        # terrain
        self.moon = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, H/8), (W, H/8)]) )
        self.moon.CreateEdgeFixture(
            vertices=[(W/8,0),(W/8,H)],
            density=0,   friction=0.1)
        self.moon.CreateEdgeFixture(
            vertices=[(0,H*7/8),(W,H*7/8)],
            density=0,
            friction=0.1)
        self.moon.CreateEdgeFixture(
            vertices=[(W*7/8,0),(W*7/8,H)],
            density=0,
            friction=0.1)

        self.moon.color1 = (0.9,0.9,0.9)
        self.moon.color2 = (0.9,0.9,0.9)

        self.robot = self.world.CreateDynamicBody(
            position = (random.randint(W*2/8, W*6/8), random.randint(H*2/8, H*6/8)),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=10/SCALE, pos=(0,0)),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.5) # 0.99 bouncy
                )
        self.robot.color1 = (1,1,0)
        self.robot.color2 = (0.1,0.1,0.1)
       
        self.ball = self.world.CreateDynamicBody(
            position = (random.randint(W*2/8, W*6/8), random.randint(H*2/8, H*6/8)),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=5/SCALE, pos=(0,0)),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.5) # 0.99 bouncy
                )
        self.ball.color1 = (0.9,0.4,0.0)
        self.ball.color2 = (0.8,0.4,0.05)
       
        self.drawlist += [self.robot]
        self.drawlist += [self.ball]
        self.drawlist += [self.moon]

        return self._step(np.array([0,0]) if self.continuous else 0)[0]
    
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))
        
        # Apply robot actions
        y_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [2,4]):
            if self.continuous:
                direction = np.sign(action[1])
                y_power = np.clip(np.abs(action[1]), 0.1,1.0)*direction
            else:
                direction = action-3
                y_power = 1.0 * direction
                    
        s_power = 0.0
        if (self.continuous and np.abs(action[0]) > 0.5) or (not self.continuous and action in [1,3]):
            if self.continuous:
                direction = np.sign(action[0])
                s_power = np.clip(np.abs(action[0]), 0.1,1.0)*direction
            else:
                direction = action-2
                s_power = 1.0 * direction
                #self.robot.ApplyLinearImpulse( (-ox*SIDE_ENGINE_POWER*s_power, -oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)
        
        powert = math.sqrt(s_power**2 + y_power**2) 
        if(powert > 1.0):
            s_power = s_power/powert
            y_power = y_power/powert

        self.robot.ApplyForceToCenter( ( s_power*10000/SCALE, y_power*10000/SCALE), True) 
       

        # Random Noise
        self.robot.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        ), True)

        self.world.Step(1.0/FPS, 6*30, 2*30)
        

        # Package world state
        pos = self.robot.position
        vel = self.robot.linearVelocity
        posball = self.ball.position
        velball = self.ball.linearVelocity
        state = [
            pos.x,
            pos.y,
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.robot.angle,
            20.0*self.robot.angularVelocity/FPS,
            posball.x,
            posball.y 
            ]
        assert len(state)==8
        

        # Calculate reward
        reward = 0
        '''
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        '''

        reward -= powert*0.30  # less fuel spent is better
        #reward -= s_power*0.03
        

        # Determine completion
        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done   = True
            reward = -100
        if not self.robot.awake:
            done   = True
            reward = +100
        return np.array(state), reward, False, {}


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)
        
        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE
        self.viewer.draw_polygon([(0,0),(W,0),(W,H),(0,H)], color=(0.15,0.40,0.15))

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                elif type(f.shape) is edgeShape:
                    #t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_line(f.shape.vertex1, f.shape.vertex2, color=obj.color1)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        '''
        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(1,1,1) )
            self.viewer.draw_polygon( [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)], color=(0.8,0.8,0) )
        '''

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class SSLSimpleNavContinuous(SSLSimpleNav):
    continuous = True

def heuristic(env, s):
    # Heuristic for:
    # 1. Testing. 
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
        elif angle_todo < -0.05: a = 3
        elif angle_todo > +0.05: a = 1
    return a

if __name__=="__main__":
    #env = SSLSimpleNav()
    env = SSLSimpleNavContinuous()
    s = env.reset()
    total_reward = 0
    steps = 0
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        env.render()
        total_reward += r
        if steps % 20 == 0 or done:
            print("Action: {}".format(a))
            print(["{:+0.2f}".format(x) for x in s])
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        #if done: break
