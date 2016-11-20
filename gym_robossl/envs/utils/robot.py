class Robot:
    self._ROBOT_RADIUS

    self.box2d_body = None
    
    def __init__(self, world, position, angle):
        self.reset() 
        reset(self, world, position, angle):


    def render(self, viewer):
        self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
        self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)

    def reset(self, world, position, angle):
        self.robot = self.world.CreateDynamicBody(
            position = position,
            angle = angle,
            fixtures = fixtureDef(
                shape=circleShape(radius=10/SCALE, pos=(0,0)),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.5) # 0.99 bouncy
                )

'''
# We control
class FriendlyRobot(Robot):


# Robots controlled by antagonistic agent 
class EnemyRobot(Robot):
'''
