import argparse
from predictor import Predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Predictor for Quantum Compilation")
    parser.add_argument("figure_of_merit", type=str, help="The figure of merit to optimize (e.g., expected_fidelity)")
    parser.add_argument("trained", type=int, help="Number of previously trained timesteps")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train")
    parser.add_argument("--device_name", type=str, default="ibm_washington", help="Target quantum device")
    parser.add_argument("--model_name", type=str, default="rl", help="Name prefix for the model")
    parser.add_argument("--save_name", type=str, default="rl", help="Name predix for saving")
    parser.add_argument("--resume_path", type=str, default="./checkpoints/checkpoint.pt", help="Name predix for loading")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity level")
    parser.add_argument("--test", action="store_true", help="Run in test mode")

    args = parser.parse_args()

    rl_pred = Predictor(
        figure_of_merit=args.figure_of_merit,
        device_name=args.device_name
    )

    rl_pred.train_model(
        timesteps=args.timesteps,
        model_name=args.model_name,
        verbose=args.verbose,
        test=args.test,
        trained=args.trained,
        save_name=args.save_name,
        resume_model_path=args.resume_path
    )
