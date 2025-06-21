import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

class GraphEnv(gymnasium.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, shape_type='circle', max_steps=50):
        super(GraphEnv, self).__init__()
        # shape_type: 'circle', 'square', or 'rect'
        assert shape_type in ('circle', 'square', 'rect'), "shape_type must be 'circle', 'square', or 'rect'"
        self.shape_type = shape_type
        # Action: x, y, number of shapes, rotation angle
        self.action_space = spaces.Box(
            low=np.array([-4.0, -4.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([4.0, 4.0, 8.99, 2 * math.pi], dtype=np.float32),
            dtype=np.float32
        )
        # Observation: num_shapes, total_area, num_overlaps, placeholder, placeholder
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.array([100.0, 500.0, 500.0, 0.0, 0.0], dtype=np.float32),
            dtype=np.float32
        )

        # shape dimensions
        self.circle_radius = 1.0
        self.square_side = 0.5
        self.rect_sides = (0.5, 1.0)
        self.container_area = 30.0
        self.max_steps = max_steps
        self.current_step = 0
        self.area = 0

        self._init_heart_outline()
        self.reset()

    def _init_heart_outline(self):
        self.x_heart1 = [-3.36, -3.3, -3.2, -3.1, -3, -2.95, -2.5, -2, -1.5, -1,
                         -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1,
                         1.5, 2, 2.5, 2.95, 3, 3.1, 3.2, 3.3, 3.36]
        self.y_heart1 = [1.487, 0.948, 0.58, 0.314, 0.1, 0, -0.685, -1.23,
                         -1.66, -2.02, -2.34, -2.47, -2.55, -2.63, -2.775,
                         -2.63, -2.55, -2.47, -2.34, -2.02, -1.66, -1.23,
                         -0.685, 0, 0.1, 0.314, 0.58, 0.948, 1.487]
        self.x_heart2 = [-3.36, -3.3, -3.1, -2.7, -2.5, -2, -1.8, -1.7,
                         -1.52, -1.2, -0.5, 0, 0.5, 1.2, 1.52, 1.7,
                         1.8, 2, 2.5, 2.7, 3.1, 3.3, 3.36]
        self.y_heart2 = [1.487, 2.04, 2.55, 3.03, 3.17, 3.37, 3.41, 3.42,
                         3.43, 3.41, 3.19, 2.7775, 3.19, 3.41, 3.43,
                         3.42, 3.41, 3.37, 3.17, 3.03, 2.55, 2.04, 1.487]

    def _in_heart(self, x, y):
        return ((((x / 3.05)**2 + (y / 2.775)**2 - 0.32)**3)
                - ((x / 2.75)**2) * ((y / 2.775)**3)) < 0

    def _is_valid_position(self, shape_poly):
        for (vx, vy) in shape_poly:
            if not self._in_heart(vx, vy):
                return False
        for existing in self.placed_shapes:
            if patches.Polygon(existing).get_path().intersects_path(
                    patches.Polygon(shape_poly).get_path()):
                return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.placed_shapes = []
        self.num_overlaps = 0
        self.tot_area_shapes = 0
        self.reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            len(self.placed_shapes),
            self.tot_area_shapes,
            self.num_overlaps,
            0.0,
            0.0
        ], dtype=np.float32)

    def _make_shape(self, x, y, angle):
        t = self.shape_type
        if t == 'circle':
            theta = np.linspace(0, 2*math.pi, 20)
            self.area = math.pi * self.circle_radius**2
            return [(x + self.circle_radius*math.cos(th), y + self.circle_radius*math.sin(th)) for th in theta]
        
        corners = []
        if t == 'square':
            s = self.square_side / 2
            corners = [(-s, -s), (s, -s), (s, s), (-s, s)]
            self.area = self.square_side**2
        elif t == 'rect':
            w, h = self.rect_sides
            corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
            self.area = w * h

        poly = []
        for dx, dy in corners:
            rx = dx*math.cos(angle) - dy*math.sin(angle)
            ry = dx*math.sin(angle) + dy*math.cos(angle)
            poly.append((x + rx, y + ry))
        return poly

    def step(self, action):
        self.current_step += 1
        x, y, num, ang = action
        x, y = np.clip([x, y], -4, 4)
        num = int(np.clip(num, 0, 8))
        ang = float(ang % (2 * math.pi))

        self.reward = 0.0
        placed = 0
        overlaps_this_step = 0

        for _ in range(num):
            poly = self._make_shape(x, y, ang)
            if self._is_valid_position(poly):
                self.placed_shapes.append(poly)
                placed += 1
                self.tot_area_shapes += self.area
                self.reward += 15.0  # Base reward for a valid placement
            else:
                overlaps_this_step += 1
                self.reward -= 1.0  # More aggressive penalty for failed attempt

        # Bonus for placing multiple shapes
        if placed > 1:
            self.reward += 400.0 * (placed - 1)

        if placed == 0:
            self.reward -= 1005.0  # Bigger penalty for wasting a step

        self.num_overlaps += overlaps_this_step

        if self.tot_area_shapes / self.container_area >= 0.2:
                self.reward += 20.5
                # self.done = True

        if self.tot_area_shapes / self.container_area >= 0.3:
            self.reward += 50.0
            # self.done = True

        if self.tot_area_shapes / self.container_area >= 0.4:
            self.reward += 100.0
            # self.done = True

        if self.tot_area_shapes / self.container_area >= 0.5:
            self.reward += 150.0
            # self.done = True

        if self.tot_area_shapes / self.container_area >= 0.6:
            self.reward += 200.0
            # self.done = True

        if self.tot_area_shapes / self.container_area >= 0.8:
            self.reward += 8000.0
            self.done = True

        if self.current_step >= self.max_steps:
            self.done = True

        return self._get_obs(), self.reward, self.done, False, {}

    def render(self, mode='human'):
        fig, ax = plt.subplots()
        ax.plot(self.x_heart1, self.y_heart1, 'pink')
        ax.plot(self.x_heart2, self.y_heart2, 'pink')
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
        for poly in self.placed_shapes:
            patch = patches.Polygon(poly, closed=True, fill=False, edgecolor='red')
            ax.add_patch(patch)
        plt.title(f'Packing {self.shape_type.capitalize()}s into a Heart')
        plt.show()
        self._figure = fig

    def save_best_attempt(self, filename="best_heart_packing.png"):
        if not hasattr(self, '_figure'):
            self.render()
        self._figure.savefig(filename)
        print(f"Saved best attempt to {filename}")



# import matplotlib.pyplot as plt
# import gymnasium
# from gymnasium import spaces
# import numpy as np
# import time, datetime
# from collections import deque
# import math
    
# class GraphEnv(gymnasium.Env):
#     metadata = {'render.modes': ['human']}
#     """Custom Environment that follows gym interface"""

#     def __init__(self):
#         super(GraphEnv, self).__init__()
#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         self.action_space = spaces.Box(
#             low=np.array([-4.0,-4.0,0]), 
#             high=np.array([4.0,4.0,8.99]), 
#             dtype=np.float32,  # Circle (x, y)
#         )
#         # Example for using image as input (channel-first; channel-last also works):
#         self.observation_space = spaces.Box(
#             low=np.zeros(5,dtype=np.float32),
#             high=np.array([100.0, 10.0,30.0,30.0, 500.0]),
#             dtype=np.float32
#         )
#     def plot(self,x_pos,y_pos,shape = 'circle'):
#         circle = plt.Circle((x_pos, y_pos), 1, color='r',fill = False)
#         plt.plot(x_pos,y_pos,'d')
#         fig = plt.gcf()
#         ax = fig.gca()
#         ax.add_patch(circle)
#         self.list_shape_x.append(x_pos)
#         self.list_shape_y.append(y_pos)
#         self.list_shape.append(shape)
        
#     def check_area(self):
#         area_tot = len(self.list_shape)*(1*math.pi)
#         if area_tot <= 30:
#             self.reward += 1
#         else:
#             self.reward -= 1
#         self.tot_area_shapes = area_tot
            

#     def check_boundary(self):
#         for i in range (len(self.list_shape)):
#             x_pos = self.list_shape_x[i]
#             y_pos = self.list_shape_y[i]
#             if ((((x_pos/3.05)**2) + ((y_pos/2.775)**2)-0.32)**3)-(((x_pos/2.75)**2))*(((y_pos/2.775)**3)) < 0:
#                 self.reward += 1
#             else: 
#                 self.reward -=1
#         # plt.savefig(str(i)+"_"+str(datetime.datetime.now)+" .png")

#     def check_collisions(self):
#         for i in range(len(self.list_shape)):
#             for j in range(len(self.list_shape)):
#                 if math.dist((self.list_shape_x[i],self.list_shape_y[i]),(self.list_shape_x[j],self.list_shape_y[j]))>=2:
#                     self.reward += 1
#                 else:
#                     self.reward -= 1
#                     self.num_overlaps +=1
    

#     def step(self, action):
#         print("[DEBUG] step() recieved action: ",action,", type: ",type(action))
#         print("[DEBUG] action shape: ",getattr(action,"shape","N/A"))
#         print(self.action_space.sample())
#         print(self.action_space.sample().shape)
#         x,y,num_circles = action
#         x, y = np.clip([x,y], -4, 4)
#         num_circles = int(np.clip(num_circles,0,8))
        
#         for i in range(num_circles):
#             self.plot(x,y)
#             self.num_shapes += 1
#             self.check_boundary()
#             self.check_collisions()
#             self.check_area()
#         self.done = True
        

#         truncated = False
        
        
#         info = {}
#         self.observation = np.array([self.num_shapes,1,30,self.tot_area_shapes,self.num_overlaps],dtype=np.float32)
#         return self.observation, self.reward, self.done, truncated, info
    
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.reward = 0
#         self.done = False
#         self.observation = None
#         # bottom of heart:
#         self.x_heart1 = [-3.36,-3.3,-3.2,-3.1,-3,-2.95,-2.5, -2 , -1.5 , -1 , -0.5 ,-0.3, -0.2, -0.1, 0 ,0.1,0.2,0.3,0.5,1,1.5,2,2.5,2.95,3,3.1,3.2,3.3,3.36]
#         self.y_heart1 = [1.487,0.948,0.58,0.314,0.1,0,-0.685, -1.23,-1.66, -2.02,-2.34,-2.47,-2.55,-2.63,-2.775,-2.63,-2.55,-2.47,-2.34,-2.02,-1.66,-1.23,-0.685,0,0.1,0.314,0.58,0.948,1.487]
#         # top of heart:
#         self.x_heart2 = [-3.36,-3.3,-3.1,-2.7,-2.5,-2,-1.8,-1.7,-1.52,-1.2,-0.5,0,0.5,1.2,1.52,1.7,1.8,2,2.5,2.7,3.1,3.3,3.36]
#         self.y_heart2 = [1.487,2.04,2.55,3.03,3.17,3.37,3.41,3.42,3.43,3.41,3.19,2.7775,3.19,3.41,3.43,3.42,3.41,3.37,3.17,3.03,2.55,2.04,1.487]
#         plt.plot(self.x_heart1,self.y_heart1)
#         plt.plot(self.x_heart2,self.y_heart2)
#         plt.xlim(-4,4)
#         plt.ylim(-4,4)
#         plt.title('HEART')
#         #lists that store data:
#         self.list_shape_x = []
#         self.list_shape_y = []
#         self.list_shape = []

        
#         # num_shapes, shape_size, in_container, area_container, tot_area_shapes, num overlaps
#         self.num_shapes = 0
#         shape_size = 1
#         # in_container = 1
#         area_container = 30.0
#         self.tot_area_shapes = 0
#         self.num_overlaps = 0
#         info = {}
    

#         self.observation = np.array([self.num_shapes,shape_size,area_container,self.tot_area_shapes,self.num_overlaps],dtype=np.float32)

#         return self.observation, info  # reward, done, info can't be included
    
#     # def render(self, mode='human'):
#     #     plt.savefig("_"+datetime.datetime.now+" .png")
