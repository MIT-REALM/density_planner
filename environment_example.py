from env.environment import Environment


def main():
    env = Environment()
    env.run()
    #save animation
    env.create_animation()


if __name__ == "__main__":
    main()
    print('done')
