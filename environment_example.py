from env.environment import Environment
import hyperparams
import numpy as np


def main():
    args = hyperparams.parse_args()
    env = Environment(args, init_time=16, end_time=27)
    env.run()
    # save animation
    # env.create_no_ego_grid_animation()
    # Create waypoints
    pos0 = env.generate_random_waypoint(time=16)
    posN = env.generate_random_waypoint(time=16+11)
    print(pos0)
    print(posN)
    xref0 = np.array([pos0[0], pos0[1], 1.372713, 7.7780137, 0.])
    xrefN = np.array([posN[0], posN[1], 0., 0., 0.])
    env.create_overlay_animation(xref0, xrefN)


if __name__ == "__main__":
    main()
    print('done')
