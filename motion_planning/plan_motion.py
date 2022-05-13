from motion_planning.utils import plot_grid
from motion_planning.simulation_objects import Environment
from motion_planning.example_objects import create_environment, create_crossing4w, create_pedRL, create_street, create_turnR, create_pedLR
import hyperparams

if __name__ == '__main__':
    args = hyperparams.parse_args()

    env = create_environment(["street", "obstacleBottom", "pedLR"], args)
    plot_grid(env, args)
    env.forward_occupancy(step_size=50)
    plot_grid(env, args)





    print("end")