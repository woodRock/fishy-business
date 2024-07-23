import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

class PSO(nn.Module):
    def __init__(self, n_particles, n_iterations, c1, c2, n_classes, n_features, w_start=0.9, w_end=0.4):
        super(PSO, self).__init__()
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.c1 = c1
        self.c2 = c2
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

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")

    def fitness(self, particles, data_loader, lambda_reg=0.01):
        total_correct = torch.zeros(self.n_particles, device=self.device)
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = torch.matmul(X_batch.unsqueeze(0), particles.transpose(1, 2))
                y_pred = torch.argmax(predictions, dim=2)
                y_true = torch.argmax(y_batch, dim=1).unsqueeze(0).expand(self.n_particles, -1)
                total_correct += torch.sum(y_pred == y_true, dim=1)
                total_samples += len(y_batch)
        
        accuracy = total_correct / total_samples
        regularization = lambda_reg * torch.norm(particles, p=2, dim=(1,2))
        return accuracy - regularization

    def update_velocity(self, iteration):
        w = self.w_start - (self.w_start - self.w_end) * iteration / self.n_iterations
        r1, r2 = torch.rand(2, self.n_particles, 1, 1, device=self.device)
        self.velocities = (w * self.velocities +
                           self.c1 * r1 * (self.pbest - self.particles) +
                           self.c2 * r2 * (self.gbest.unsqueeze(0) - self.particles))

    def fit(self, train_loader, val_loader, patience=10):
        best_val_accuracy = float('-inf')
        patience_counter = 0
        
        for iteration in (pbar := tqdm(range(self.n_iterations), desc="Training PSO Classifier")):
            with torch.no_grad():
                # Update velocities and positions
                self.update_velocity(iteration)
                self.particles += self.velocities
                
                # Evaluate fitness for all particles
                fitness_values = self.fitness(self.particles, train_loader)
                
                # Update personal and global best
                improved_particles = fitness_values > self.pbest_fitness
                self.pbest_fitness[improved_particles] = fitness_values[improved_particles]
                self.pbest[improved_particles] = self.particles[improved_particles].clone()
                
                best_particle = torch.argmax(fitness_values)
                if fitness_values[best_particle] > self.gbest_fitness:
                    self.gbest_fitness = fitness_values[best_particle]
                    self.gbest = self.particles[best_particle].clone()
                
                # Training accuracy
                train_accuracy = self.evaluate(train_loader)
                
                # Evaluate on validation set
                val_accuracy = self.evaluate(val_loader)
                
                if train_accuracy == 1:
                    # Early stopping check
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at iteration {iteration+1}")
                        break
                
                # Logging
                message = f"Iteration {iteration+1}/{self.n_iterations}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
                self.logger.info(message)
                pbar.set_description(message)

    def predict(self, data_loader):
        all_predictions = []
        self.gbest = self.gbest.to(self.device)
        
        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                predictions = X_batch @ self.gbest.T
                y_pred = torch.argmax(predictions, dim=1)
                all_predictions.append(y_pred.cpu())
        
        return torch.cat(all_predictions).numpy()
    
    def evaluate(self, data_loader):
        all_predictions = []
        all_true_labels = []
        self.gbest = self.gbest.to(self.device)

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                predictions = x @ self.gbest.T
                y_pred = torch.argmax(predictions, dim=1)
                y_true = torch.argmax(y, dim=1)
                
                all_predictions.extend(y_pred.cpu().numpy())
                all_true_labels.extend(y_true.cpu().numpy())

        balanced_accuracy = balanced_accuracy_score(all_true_labels, all_predictions)
        return balanced_accuracy