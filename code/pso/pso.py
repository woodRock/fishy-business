import torch

class PSO:
    def __init__(self, n_particles, n_iterations, c1, c2, w, n_classes, n_features):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.n_classes = n_classes
        self.n_features = n_features
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize particles and velocities
        self.particles = torch.randn(n_particles, n_classes, n_features, device=self.device)
        self.velocities = torch.zeros_like(self.particles)
        
        # Initialize personal and global best
        self.pbest = self.particles.clone()
        self.pbest_fitness = torch.full((n_particles,), float('-inf'), device=self.device)
        self.gbest = self.particles[0].clone()
        self.gbest_fitness = torch.tensor(float('-inf'), device=self.device)

    def fitness(self, particle, data_loader):
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
        return total_correct / total_samples

    def update_position(self, particle, velocity):
        return particle + velocity

    def update_velocity(self, particle, velocity, pbest, gbest):
        r1, r2 = torch.rand(2, device=self.device)
        new_velocity = (self.w * velocity +
                        self.c1 * r1 * (pbest - particle) +
                        self.c2 * r2 * (gbest - particle))
        return new_velocity

    def fit(self, train_loader, val_loader):
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                fitness = self.fitness(self.particles[i], train_loader)
                
                # Update personal best
                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest[i] = self.particles[i].clone()
                
                # Update global best
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest = self.particles[i].clone()
            
            # Update velocities and positions
            for i in range(self.n_particles):
                self.velocities[i] = self.update_velocity(self.particles[i], self.velocities[i], self.pbest[i], self.gbest)
                self.particles[i] = self.update_position(self.particles[i], self.velocities[i])
            
            # Evaluate on validation set
            val_accuracy = self.fitness(self.gbest, val_loader)
            print(f"Iteration {iteration+1}/{self.n_iterations}, Training Accuracy {self.gbest_fitness:.4f} Validation Accuracy: {val_accuracy:.4f}")

    def predict(self, data_loader):
        all_predictions = []
        self.gbest = self.gbest.to(self.device)
        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                y_pred = torch.argmax(X_batch @ self.gbest.T, dim=1)
                all_predictions.append(y_pred.cpu())
        return torch.cat(all_predictions).numpy()