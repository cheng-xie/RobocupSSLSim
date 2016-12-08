class FullField:
    self.box2d_bounds = None
    def __init__(self, W, H, margin):
        self.rgbfield = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, 0), (W, 0)]) )

    def reset_border(W, H, margin): 
        self.rgbfield.CreateEdgeFixture(
            vertices=[(0,margin),(W,margin)],
            density=0,
            friction=0.1)
        self.rgbfield.CreateEdgeFixture(
            vertices=[(0,H-margin),(W,H-margin)],
            density=0,
            friction=0.1)
        self.rgbfield.CreateEdgeFixture(
            vertices=[(0,H/4),(W,H/4)],
            density=0,
            friction=0.1)
        self.rgbfield.CreateEdgeFixture(
            vertices=[(0,H/4),(W,H/4)],
            density=0,
            friction=0.1)
    

    def destroy():

    def check_border(self, r):
      
    def render():
        # draw the background

        # draw lines for the boundaries
        

        # field lines

class BoundaryEdge:
    def __init__(start, end, color):
        self.start = start 
        self.end = end 
        self.color = color
        
    def render(viewer):
        viewer.draw_line(start,end,color = color)
