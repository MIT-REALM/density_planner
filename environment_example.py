from env.environment import Environment
import hyperparams

def main():
    args = hyperparams.parse_args()
    env = Environment(args, init_time=0, end_time=40)
    env.run()
    #save animation
    env.create_animation()


if __name__ == "__main__":
    main()
    print('done')
