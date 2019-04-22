import visdom
import numpy as np
vis = visdom.Visdom(server='172.220.4.32', port='6006')
vis.text('Hello, world!')
vis.image(np.ones((3, 10, 10)))
