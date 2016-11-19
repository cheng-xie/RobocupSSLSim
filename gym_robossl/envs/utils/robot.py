class Robot:
    self._ROBOT_RADIUS

    self.box2d_body = None
    def __init__(self):
        self.box2d_body = fixtureDef(
                shape=circleShape(radius=self._ROBOT_RADIUS, pos=(0,0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
    def draw():
    
    def reset():

# We control
class FriendlyRobot(Robot):

# Robots controlled by antagonistic agent 
class EnemyRobot(Robot):
