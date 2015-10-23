from vsip.parallel import *
from vsip.parallel.map import map
from vsip.parallel.block import block
from vsip import matrix
import matplotlib.pyplot as plt
from matplotlib.colors import *
import matplotlib.patches as mpatches

SIZE=16

def iterate(m):

    for x in xrange(m.cols()):
        for y in xrange(m.rows()):
            yield (y, x)


colormap = ['#ff0000', #red
            '#00ff00', #green
            '#0000ff', #blue
            '#ffff00', #yellow
            '#ff00ff', #magenta
            '#00ffff', #cyan
            '#ffffff'] #white

def plot(mat, proc=None, dist=None):
    """Print matrix 'mat' either for processor 'proc' or distribution 'dist'."""

    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    if proc is not None:
        plt.title('local submatrix of processor %d'%proc)
    else:
        plt.title('distributed matrix')
        patches = []
        for proc in xrange(num_processors()):
            c = colormap[proc%len(colormap)]
            patches.append(mpatches.Patch(color=c, label='processor %d'%proc))
        # Shink current axis by 40%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
        # Put a legend to the right of the current axis
        ax.legend(handles=patches, bbox_to_anchor=(1.6,1))
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        
    for (y,x) in iterate(mat):
        if dist:
            proc = dist.block.map().owner((y, x))
        # color the cells according to their locality
        c = colormap[proc%len(colormap)]
        # translate color into HSV triple
        c = rgb_to_hsv(hex2color(colormap[proc%len(colormap)]))
        # scale the 'value' part
        c = (c[0], c[1], c[2] * (.75 + mat[y,x] / 1024))
        # translate color back into hex string
        c = rgb2hex(hsv_to_rgb(c))
        rect = plt.Rectangle([x - 0.5, y - 0.5], 1, 1, facecolor=c)
        ax.add_patch(rect)
        ax.annotate(str(mat[y,x]), (x, y), color='white', weight='bold', 
                    fontsize=6, ha='center', va='center')
        
    ax.autoscale_view()
    ax.invert_yaxis()


# Create a map...
if num_processors() > 1:
    nr = num_processors() // 2
    nc = num_processors() // nr
    m = map(nr, nc)
else:
    m = map(1)
comm = m.communicator()

# ...and a distributed matrix with it...
m = matrix(block=block(int, (SIZE,SIZE), 0., m))
# ...as well as a local one in the rank=1 process
lm = matrix(block=block(int, (SIZE,SIZE), 0., map(1)))

# Now manipulate the distributed matrix...
for (y, x) in iterate(m):
    m[y, x] = x + y * m.cols()

# ...and assign it to local.
lm[:,:] = m

# Now print all the local (sub-)matrices...
f1 = plt.figure(1)
plot(m.local(), proc=local_processor())
f1.show()
# ...as well as the distributed matrix
if local_processor() == 0:
    f2 = plt.figure(2)
    plot(lm, dist=m)
    f2.show()
    raw_input()
    comm.barrier()
else:
    # Wait for the main process to finish
    comm.barrier()
