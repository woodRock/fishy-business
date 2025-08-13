# Replace the entire contents of siamese/gp.py
"""
Final, definitive version of the multi-gene Genetic Programming algorithm.
Implements early stopping and bloat control to combat overfitting.
"""
import multiprocessing
import random
import os
import sys
import numpy as np
import functools
from sklearn.metrics import balanced_accuracy_score
import torch
from typing import List, Tuple
from deap import base, creator, gp, tools, algorithms

# Correctly use a relative import for a package
from .util import prepare_dataset, DataConfig


# Helper function to convert DataLoader to a list
def flatten_dataloader(
    data_loader: torch.utils.data.DataLoader,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    data = []
    for batch in data_loader:
        sample1, sample2, labels = batch
        data.extend(list(zip(sample1, sample2, labels)))
    return data


def setup_primitives(n_inputs: int) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", n_inputs)
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    # pset.addPrimitive(lambda x, y: np.divide(x, y, out=np.ones_like(x), where=abs(y)>1e-6), 2, "pdiv")
    pset.addPrimitive(np.tanh, 1, name="tanh")
    pset.addPrimitive(np.sin, 1, name="sin")
    pset.addPrimitive(np.cos, 1, name="cos")
    pset.addEphemeralConstant("rand", functools.partial(random.uniform, -1, 1))
    return pset


def get_embeddings(individual: list, data: list) -> Tuple[np.ndarray, np.ndarray, list]:
    outputs1, outputs2, true_labels = [], [], []
    for sample1, sample2, label_tensor in data:
        s1, s2 = sample1.numpy(), sample2.numpy()
        e1 = np.array([gene(*s1) for gene in individual])
        e2 = np.array([gene(*s2) for gene in individual])
        outputs1.append(e1)
        outputs2.append(e2)
        true_labels.append(label_tensor.argmax().item())

    outputs1 = np.nan_to_num(np.array(outputs1), nan=0.0, posinf=1e6, neginf=-1e6)
    outputs2 = np.nan_to_num(np.array(outputs2), nan=0.0, posinf=1e6, neginf=-1e6)
    return outputs1, outputs2, true_labels


def calculate_accuracy(individual: list, data: list) -> float:
    if not data:
        return 0.0
    outputs1, outputs2, true_labels = get_embeddings(individual, data)

    if np.std(outputs1) < 1e-6 or np.std(outputs2) < 1e-6:
        return 0.5

    eps = 1e-8
    norm1 = np.linalg.norm(outputs1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(outputs2, axis=1, keepdims=True)
    o1_norm = np.divide(outputs1, norm1, out=np.zeros_like(outputs1), where=norm1 > eps)
    o2_norm = np.divide(outputs2, norm2, out=np.zeros_like(outputs2), where=norm2 > eps)

    sims = np.clip((np.sum(o1_norm * o2_norm, axis=1) + 1) / 2, 0, 1)
    scores = [
        balanced_accuracy_score(true_labels, sims > t) for t in np.linspace(0, 1, 50)
    ]
    return max(scores)


def calculate_loss(individual: list, data: list) -> tuple[float]:
    try:
        if not data:
            return (float("inf"),)
        loss_fn = torch.nn.ContrastiveLoss(margin=1.0)
        total_loss = 0.0

        eval_data = random.sample(data, k=min(len(data), 500))

        for sample1, sample2, label_tensor in eval_data:
            label = 1 if label_tensor.argmax().item() == 1 else -1
            target = torch.tensor([label], dtype=torch.float)
            s1_np, s2_np = sample1.numpy(), sample2.numpy()

            emb1 = torch.tensor(
                [gene(*s1_np) for gene in individual], dtype=torch.float32
            )
            emb2 = torch.tensor(
                [gene(*s2_np) for gene in individual], dtype=torch.float32
            )

            emb1, emb2 = torch.nan_to_num(emb1), torch.nan_to_num(emb2)

            if torch.std(emb1) < 1e-6 or torch.std(emb2) < 1e-6:
                return (float("inf"),)

            # Contrastive loss in PyTorch uses distance, not similarity
            # So we add the loss directly without special handling
            total_loss += loss_fn(emb1.unsqueeze(0), emb2.unsqueeze(0), target)

        # Bloat Control: Add a penalty for overly complex trees
        total_size = sum(len(gene) for gene in individual)
        size_penalty = total_size * 0.0001  # Small penalty factor

        return ((total_loss.item() / len(eval_data)) + size_penalty,)
    except Exception:
        return (float("inf"),)


def eval_wrapper_for_map(
    individual: list, data: list, pset: gp.PrimitiveSet
) -> tuple[float]:
    compiled_genes = [gp.compile(expr=gene, pset=pset) for gene in individual]
    return calculate_loss(compiled_genes, data)


def cxIndividual(ind1: list, ind2: list) -> tuple[list, list]:
    idx = random.randint(0, len(ind1) - 1)
    ind1[idx], ind2[idx] = gp.cxOnePoint(ind1[idx], ind2[idx])
    return ind1, ind2


def mutIndividual(ind: list, pset: gp.PrimitiveSet, indpb: float) -> tuple[list]:
    for i in range(len(ind)):
        if random.random() < indpb:
            expr = functools.partial(gp.genFull, pset=pset, min_=1, max_=3)
            (ind[i],) = gp.mutUniform(ind[i], expr=expr, pset=pset)
    return (ind,)


def main():
    # --- Configuration ---
    data_file_path = "/Users/woodj/Desktop/fishy-business/data/REIMS.xlsx"
    N_OUTPUTS, POP_SIZE, N_GENS = 16, 250, 100
    CXPB, MUTPB, MUT_INDPB = 0.8, 0.2, 0.1
    data_config = DataConfig(batch_size=32, data_path=data_file_path)

    # --- Data Loading ---
    train_loader, val_loader = prepare_dataset(data_config)
    N_INPUTS = next(iter(train_loader))[0].shape[1]
    train_data, val_data = flatten_dataloader(train_loader), flatten_dataloader(
        val_loader
    )
    print(
        f"Data ready. Train pairs: {len(train_data)}, Validation pairs: {len(val_data)}\n"
    )

    # --- DEAP Setup ---
    pset = setup_primitives(n_inputs=N_INPUTS)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.tree, n=N_OUTPUTS
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", cxIndividual)
    toolbox.register("mutate", mutIndividual, pset=pset, indpb=MUT_INDPB)

    # --- Evolution ---
    with multiprocessing.Pool() as pool:
        evaluate_for_pool = functools.partial(eval_wrapper_for_map, pset=pset)
        toolbox.register("map", pool.map)

        print(f"--- Starting Evolution (Minimizing Loss) ---")
        pop = toolbox.population(n=POP_SIZE)

        # Early Stopping variables
        best_val_acc = 0.0
        best_ind_ever = None

        for gen in range(N_GENS):
            eval_data_subset = random.sample(train_data, k=min(len(train_data), 500))
            fitnesses = toolbox.map(
                functools.partial(evaluate_for_pool, data=eval_data_subset), pop
            )
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # --- Reporting and Early Stopping Step ---
            # Find the best individual of the current generation based on loss
            current_best_ind = tools.selBest(pop, 1)[0]

            # Compile it once for accuracy reporting
            compiled_best = [
                gp.compile(expr=gene, pset=pset) for gene in current_best_ind
            ]

            train_acc = calculate_accuracy(compiled_best, train_data)
            val_acc = calculate_accuracy(compiled_best, val_data)
            min_loss = current_best_ind.fitness.values[0]

            print(
                f"Gen {gen:02d}: Min Loss={min_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}"
            )

            # Check if this is the best validation accuracy we've seen so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_ind_ever = toolbox.clone(current_best_ind)
                print(f"    ⭐ New best validation accuracy found: {best_val_acc:.4f}")

            # --- Standard evolutionary algorithm steps ---
            offspring = toolbox.select(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for i in range(1, len(offspring), 2):
                if random.random() < CXPB:
                    offspring[i - 1], offspring[i] = toolbox.mate(
                        offspring[i - 1], offspring[i]
                    )
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < MUTPB:
                    (offspring[i],) = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            pop[:] = offspring

    print("--- Evolution Finished ---\n")

    # Final evaluation using the best model found via early stopping
    print(f"🏆 Best validation accuracy during run: {best_val_acc:.4f}")
    compiled_final_best = [gp.compile(expr=gene, pset=pset) for gene in best_ind_ever]
    final_val_acc = calculate_accuracy(compiled_final_best, val_data)
    print(f"🏆 Final Accuracy of Best Model on Validation Set: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
