import logging
import argparse
import operator
import os
import numpy as np
from deap import base, creator, tools, gp
from deap.gp import PrimitiveSetTyped
from sklearn.model_selection import StratifiedKFold
from util import compileMultiTree, evaluate_classification
from operators import xmate, xmut, staticLimit
from gp import train, save_model, load_model
from data import load_dataset
from plot import plot_tsne, plot_gp_tree

# Disable the warnings.
# Source: https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
import warnings

warnings.filterwarnings("ignore")


def parse_arguments():
    # Handle the command line arguments for the script.
    parser = argparse.ArgumentParser(
        prog="Embedded Genetic Programming",
        description="An embedded GP for fish species classification.",
        epilog="Implemented in deap and written in python.",
    )
    parser.add_argument(
        "-f",
        "--file-path",
        type=str,
        default="checkpoints/embedded-gp.pth",
        help="The filepath to store the checkpoints. Defaults to checkpoints/embedded-gp.pth",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="species",
        help="The fish species or part dataset. Defaults to species.",
    )
    parser.add_argument(
        "-l",
        "--load",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="To load a checkpoint from a file. Defaults to false",
    )
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        default=0,
        help="The number for the run, this effects the random seed. Defaults to 0",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"logs/results",
        help="Partial filepath for the output logging. Defaults to 'logs/results'.",
    )
    parser.add_argument(
        "-p",
        "--population",
        type=int,
        default=1023,
        help="The number of individuals in the population. Defaults to 1023.",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=int,
        default=-1,
        help="Specify beta * num_features as population size. Defaults to -1.",
    )
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=10,
        help="The number of generations, or epochs, to train for. Defaults to 10.",
    )
    parser.add_argument(
        "-mx",
        "--mutation-rate",
        type=float,
        default=0.2,
        help="The probability of a mutation operations occuring. Defaults to 0.2",
    )
    parser.add_argument(
        "-cx",
        "--crossover-rate",
        type=int,
        default=0.8,
        help="The probability of a mutation operations occuring. Defaults to 0.2",
    )
    parser.add_argument(
        "-e",
        "--elitism",
        type=int,
        default=0.1,
        help="The ratio of elitists to be kept each generation.",
    )
    parser.add_argument(
        "-td",
        "--tree-depth",
        type=int,
        default=6,
        help="The maximum tree depth for GP trees. Defaults to 6.",
    )

    return parser.parse_args()


def setup_logging(args):
    logger = logging.getLogger(__name__)
    output = f"{args.output}_{args.run}.log"
    logging.basicConfig(filename=output, level=logging.INFO, filemode="w")
    return logger


def main():
    args = parse_arguments()
    logger = setup_logging(args)

    # Freeze the random seed for reproducability.
    np.random.seed(args.run)

    n_features = 1023
    if args.dataset == "instance-recognition":
        n_features = 2046
    n_classes_per_dataset = {
        "species": 2,
        "part": 6,
        "oil": 7,
        "cross-species": 3,
        "cross-species-hard": 15,
        "instance-recognition": 2,
    }

    if args.dataset not in n_classes_per_dataset:
        raise ValueError(
            f"Invalid dataset: {args.dataset} not in {n_classes_per_dataset.keys()}"
        )

    n_classes = n_classes_per_dataset[args.dataset]

    assert (
        args.crossover_rate + args.mutation_rate == 1
    ), "Crossover and mutation sums to 1 (to please the Gods!)"

    X, y = load_dataset(dataset=args.dataset)

    # Terminal set.
    # pset = gp.PrimitiveSet("MAIN", n_features)
    pset = PrimitiveSetTyped("main", [float] * n_features, float)

    def protectedDiv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.divide(
            left, right, out=np.ones_like(left, dtype=float), where=right != 0
        )

    def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x.astype(float) + y.astype(float)

    def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x.astype(float) - y.astype(float)

    def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x.astype(float) * y.astype(float)

    def neg(x: np.ndarray) -> np.ndarray:
        return -x.astype(float)

    # Basic arithmetic
    pset.addPrimitive(protectedDiv, [float, float], float, name="/")
    pset.addPrimitive(add, [float, float], float, name="+")
    pset.addPrimitive(mul, [float, float], float, name="x")
    pset.addPrimitive(sub, [float, float], float, name="-")
    pset.addPrimitive(neg, [float], float, name="-1*")

    # Trigonometry
    # pset.addPrimitive(np.sin, [float], float, name="sin")
    # pset.addPrimitive(np.cos, [float], float, name="cos")
    # pset.addPrimitive(np.tan, [float], float, name="tan")
    # pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

    toolbox = base.Toolbox()

    minimized = False
    if minimized:
        weight = -1.0
    else:
        weight = 1.0

    weights = (weight,)

    if minimized:
        creator.create("FitnessMin", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMin)
    else:
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # MCIFC constructs 8 feautres for a (c=4) multi-class classification problem (Tran 2019).
    # c - number of classes, r - construction ratio, m - total number of constructed features.
    # m = r * c = 2 ratio * 4 classes = 8 features

    k = 3 if args.dataset == "part" or args.dataset == "cross-species-hard" else 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        r = 1
        c = n_classes
        m = r * c

        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.expr, n=m
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", compileMultiTree, X=X_train)

        toolbox.register(
            "evaluate",
            evaluate_classification,
            toolbox=toolbox,
            pset=pset,
            X=X_train,
            y=y_train,
        )
        toolbox.register("select", tools.selTournament, tournsize=7)
        toolbox.register("mate", xmate)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", xmut, expr=toolbox.expr_mut, pset=pset)

        # See https://groups.google.com/g/deap-users/c/pWzR_q7mKJ0
        toolbox.decorate(
            "mate",
            staticLimit(key=operator.attrgetter("height"), max_value=args.tree_depth),
        )
        toolbox.decorate(
            "mutate",
            staticLimit(key=operator.attrgetter("height"), max_value=args.tree_depth),
        )

        # File path for saved model.
        pop, log, hof = None, None, None

        # If a saved model exists?
        if args.load and os.path.isfile(args.file_path):
            s = f"Loading model from file: {args.file_path}"
            logger.info(s)
            print(s)
            pop, log, hof = load_model(
                file_path=args.file_path, toolbox=toolbox, generations=10
            )
        else:
            s = f"No model found. Train from scratch."
            logger.info(s)
            print(s)
            pop, log, hof = train(
                generations=args.generations,
                population=args.population,
                elitism=args.elitism,
                crossover_rate=args.crossover_rate,
                mutation_rate=args.mutation_rate,
                run=args.run,
                toolbox=toolbox,
            )

        logger.info(f"Saving model to file: {args.file_path}")
        save_model(
            file_path=args.file_path,
            population=pop,
            generations=args.generations,
            hall_of_fame=hof,
            toolbox=toolbox,
            logbook=log,
            run=args.run,
        )  # Best accuracy: 0.911423

        best = hof[0]
        features = toolbox.compile(expr=best, pset=pset, X=X_train)
        evaluate_classification(
            best, toolbox=toolbox, pset=pset, verbose=True, X=X_train, y=y_train
        )
        evaluate_classification(
            best, toolbox=toolbox, pset=pset, verbose=True, X=X_test, y=y_test
        )
        # plot_tsne(dataset=args.dataset, X=X, y=y, features=features, toolbox=toolbox)
        # plot_gp_tree(best)


if __name__ == "__main__":
    main()
