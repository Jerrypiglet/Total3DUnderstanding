def clip(subjectPolygon, clipPolygon):
   # https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0]
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return ((n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3)

   outputList = subjectPolygon
   cp1 = clipPolygon[-1]

   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]

      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
   return(outputList)

# https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def vis_cube_plt(Xs, ax, color=None):
    index1 = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]
    index2 = [[1, 5], [2, 6], [3, 7]]
#     ax.scatter3D(Xs[:, 0], Xs[:, 1], Xs[:, 2])
    if color is None:
        color = list(np.random.choice(range(256), size=3) / 255.)
        print(color)
    ax.plot3D(Xs[index1, 0], Xs[index1, 1], Xs[index1, 2], color=color)
    for index in index2:
        ax.plot3D(Xs[index, 0], Xs[index, 1], Xs[index, 2], color=color)

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
def vis_axis(ax):
    for vec, tag, tag_loc in zip([([0, 1], [0, 0], [0, 0]), ([0, 0], [0, 1], [0, 0]), ([0, 0], [0, 0], [0, 1])], [r'$X_w$', r'$Y_w$', r'$Z_w$'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        a = Arrow3D(vec[0], vec[1], vec[2], mutation_scale=20,
                lw=1, arrowstyle="->", color="k")
        ax.text3D(tag_loc[0], tag_loc[1], tag_loc[2], tag)
        ax.add_artist(a)

def vis_axis_xyz(ax, x, y, z, origin=[0., 0., 0.], suffix='_w'):
    for vec, tag, tag_loc in zip([([origin[0], (origin+x)[0]], [origin[1], (origin+x)[1]], [origin[2], (origin+x)[2]]), \
       ([origin[0], (origin+y)[0]], [origin[1], (origin+y)[1]], [origin[2], (origin+y)[2]]), \
          ([origin[0], (origin+z)[0]], [origin[1], (origin+z)[1]], [origin[2], (origin+z)[2]])], [r'$X%s$'%suffix, r'$Y%s$'%suffix, r'$Z%s$'%suffix], [origin+x, origin+y, origin+z]):
        a = Arrow3D(vec[0], vec[1], vec[2], mutation_scale=20,
                lw=1, arrowstyle="->", color="k")
        ax.text3D(tag_loc[0], tag_loc[1], tag_loc[2], tag)
        ax.add_artist(a)