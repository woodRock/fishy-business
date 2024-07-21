import logging
import torch
import torch.nn as nn
from tqdm import tqdm 


class PSO(nn.Module):
    def __init__(self, n_particles, n_iterations, c1, c2, w, n_classes, n_features, w_start=0.9, w_end=0.4):
        super(PSO, self).__init__()
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.n_classes = n_classes
        self.n_features = n_features
        self.w_start = w_start
        self.w_end = w_end

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize particles and velocities
        self.particles = torch.randn(n_particles, n_classes, n_features, device=self.device)
        self.velocities = torch.zeros_like(self.particles)
        
        # Initialize personal and global best
        self.pbest = self.particles.clone()
        self.pbest_fitness = torch.full((n_particles,), float('-inf'), device=self.device)
        self.gbest = self.particles[0].clone()
        self.gbest_fitness = torch.tensor(float('-inf'), device=self.device)

        logger = logging.getLogger(__name__)
        logger.info(f"Using device: {self.device}")

    def fitness(self, particle, data_loader, lambda_reg=0.01):
        total_correct = 0
        total_samples = 0
        particle = particle.to(self.device)
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = torch.argmax(X_batch @ particle.T, dim=1)
                y_true = torch.argmax(y_batch, dim=1)
                total_correct += torch.sum(y_pred == y_true).item()
                total_samples += len(y_batch)
        accuracy = total_correct / total_samples
        # regularization = lambda_reg * torch.norm(particle, p=2)
        # return accuracy - regularization
        return accuracy

    def update_position(self, particle, velocity):
        return particle + velocity

    def update_velocity(self, particle, velocity, pbest, gbest, iteration):
        self.w = self.w_start - (self.w_start - self.w_end) * iteration / self.n_iterations
        r1, r2 = torch.rand(2, device=self.device)
        new_velocity = (self.w * velocity +
                        self.c1 * r1 * (pbest - particle) +
                        self.c2 * r2 * (gbest - particle))
        return new_velocity

    def fit(self, train_loader, val_loader, patience=10):
        logger = logging.getLogger(__name__)
        best_val_accuracy = float('-inf')
        patience_counter = 0
        
        with torch.no_grad():
            for iteration in (pbar := tqdm(range(self.n_iterations), desc="Training PSO Classifier")):
                # Vectorized update
                r1, r2 = torch.rand(2, 1, 1, device=self.device)
                self.velocities = (self.w * self.velocities +
                                self.c1 * r1 * (self.pbest - self.particles) +
                                self.c2 * r2 * (self.gbest - self.particles))
                self.particles += self.velocities
                
                # Fitness evaluation
                fitness_values = torch.tensor([self.fitness(p, train_loader) for p in self.particles])
                
                # Update personal and global best
                # Move to device 
                fitness_values, self.pbest_fitness = fitness_values.to(self.device), self.pbest_fitness.to(self.device)
                improved_particles = fitness_values > self.pbest_fitness
                self.pbest_fitness[improved_particles] = fitness_values[improved_particles]
                self.pbest[improved_particles] = self.particles[improved_particles].clone()
                
                best_particle = torch.argmax(fitness_values)
                if fitness_values[best_particle] > self.gbest_fitness:
                    self.gbest_fitness = fitness_values[best_particle]
                    self.gbest = self.particles[best_particle].clone()
                
                # Evaluate on validation set
                val_accuracy = self.fitness(self.gbest, val_loader)
                
                if self.gbest_fitness >= 1:
                    # Early stopping check
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at iteration {iteration+1}")
                        break
                
                # Logging
                message = f"Iteration {iteration+1}/{self.n_iterations}, Training Accuracy {self.gbest_fitness:.4f} Validation Accuracy: {val_accuracy:.4f}"
                logger.info(message)
                pbar.set_description(message)

    def predict(self, data_loader):
        all_predictions = []
        self.gbest = self.gbest.to(self.device)
        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                y_pred = torch.argmax(X_batch @ self.gbest.T, dim=1)
                all_predictions.append(y_pred.cpu())
        return torch.cat(all_predictions).numpy()