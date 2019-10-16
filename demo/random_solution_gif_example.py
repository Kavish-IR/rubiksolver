from rubiksolver import *
import copy

c = Cube()

c.start_recorder()
c.randomize_state(seed=1984)
initial_scramble = c.recorded_moves
c.flush_recorder()

s = Solver(c)
s.solve()

c2 = Cube()
c2.perform_move_list(initial_scramble)

fig = plt.figure(figsize=(8,8))
ax  = fig.add_subplot(111, projection='3d')

def save_fig(c, ax, title_str, i):
    print(i, title_str)
    ax = c.cube_plot(ax=ax, title_str=title_str)
    plt.savefig('./tmp/fig_{0:03}.png'.format(i), dpi=50)    
    return ax

ax = save_fig(c2, ax, "Initial State", 0)
ax.clear()

for i, move in enumerate(c.recorded_moves):
    c2.perform_move(move)
    ax = save_fig(c2, ax, "Move {0:03}: {1}".format(i+1, move), i+1)
    ax.clear()

ax = save_fig(c2, ax, "Final State", len(c.recorded_moves)+2)
ax.clear()

## Stitch result png to gif with the following imagemagick command :
# convert -delay 1x1 fig_000.png -delay 1x4 fig_*.png -delay 1x1 fig_143.png -loop 0 solution.gif
## then used the following to rescale
# convert ../img/solution.gif -coalesce -scale 300x300 -fuzz 2% +dither -remap ../img/solution.gif[0] -layers Optimize solution_example.gif

