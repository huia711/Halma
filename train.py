from ppo import Config, PPOTrainer


def main():
    hyp = Config()
    trainer = PPOTrainer(hyp)
    trainer.train()


if __name__ == '__main__':
    main()
