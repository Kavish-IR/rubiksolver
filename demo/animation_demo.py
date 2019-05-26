from Solver import *
from Cube   import *

# Fresh cube, set to random state:
c = Cube()
c.randomize_state()
initial_state = c.export_state()

# Solve and retain moves
s = Solver(c)
s.solve()
solve_moves = c.recorded_moves

# Place cube back in initial_state
c.set_state(initial_state)

# Plot initial state
plt.ion()
ax  = c.cube_plot()
axs = c.square_plot()
fig = plt.gcf()
plt.pause(0.0125)

# Walk through the solution
for move in solve_moves:
    c.perform_move(move)
    c.cube_plot(ax=ax)
    c.square_plot(axs=axs)
    plt.draw()
    plt.pause(0.0125)

# Display final state
plt.ioff()
c.cube_plot(ax=ax)
c.square_plot(axs=axs)
plt.show()
