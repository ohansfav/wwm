# show Python and architecture
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn_extra.cluster import KMedoids 

# --- 1. Define the Fitness Function (Objective Function) ---
# (This function remains the same as before)
def calculate_fitness(solution, X_train, X_test, y_train, y_test):
    """
    Calculates the fitness of a binary solution (feature subset).
    Fitness is based on classification accuracy and the number of selected features.
    """
    selected_indices = np.where(solution == 1)[0] # Ensure we get indices correctly
    if len(selected_indices) == 0:
        return 0, 0 

    # Select features using advanced indexing
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, predictions)
    num_features = len(selected_indices)
    total_features = X_train.shape[1] # Use shape[1] for total features

    alpha = 0.9 # Weight for accuracy
    fitness_score = alpha * accuracy + (1 - alpha) * (1 - (num_features / total_features))

    return fitness_score, accuracy

# --- 2. Implement the Binary Waterwheel Plant Algorithm (bWWPA) ---

def binary_waterwheel_plant_algorithm(X, y, n_plants, max_iter, alpha=0.9):
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    positions = np.random.randint(0, 2, size=(n_plants, n_features))
    # Note: the fitness scores tuple structure requires modification for np.argmax
    # We will compute fitness on the fly or store just the score part
    best_position = positions[0].copy()
    best_fitness = -1.0
    
    K = 1.0 
    
    print(f"\nStarting Optimization with {n_plants} plants and {max_iter} iterations...")

    for i in range(n_plants):
         current_fitness, _ = calculate_fitness(positions[i], X_train, X_test, y_train, y_test)
         if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_position = positions[i].copy()

    print(f"Initial Best Fitness: {best_fitness:.4f}")

    for t in range(1, max_iter + 1):
        for i in range(n_plants):
            r = np.random.rand()
            r1, r2, r3 = np.random.rand(3)
            
            current_position = positions[i].copy()

            if r < 0.5:
                W_vec = r1 * (current_position + 2 * K)
                new_position = current_position + W_vec * (2 * K + r2)
            else:
                W_vec = r3 * (K * best_position + r3 * current_position)
                new_position = current_position + K * W_vec
            
            sigmoid_val = 1 / (1 + np.exp(-new_position))
            positions[i] = (sigmoid_val > np.random.rand(n_features)).astype(int)

            current_fitness, _ = calculate_fitness(positions[i], X_train, X_test, y_train, y_test)

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_position = positions[i].copy()

        K = 1 + 2 * t**2 / (max_iter)**3 + 0.5 

        if t % 10 == 0:
            print(f"Iteration {t}/{max_iter}, Best Fitness so far: {best_fitness:.4f}")

    _, best_accuracy = calculate_fitness(best_position, X_train, X_test, y_train, y_test)
    selected_indices = np.where(best_position == 1)[0]
    
    print("-" * 30)
    print(f"Optimization finished.")
    print(f"Final Best Accuracy: {best_accuracy:.4f}")
    print(f"Selected Features Count: {len(selected_indices)}/{n_features}")
    print(f"Selected Feature Indices: {selected_indices}")
    print("-" * 30)
    
    return best_position, selected_indices

# --- 3. Main execution with User Input ---

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    data = load_iris()
    X = data.data
    y = data.target

    # Get user input for parameters
    try:
        n_plants_input = int(input("Enter the number of waterwheel plants (e.g., 20): "))
        max_iter_input = int(input("Enter the maximum number of iterations (e.g., 50): "))
    except ValueError:
        print("Invalid input. Please enter integers for the parameters.")
        exit() # Exit the program if input is invalid

    # Run the algorithm with user-specified inputs
    selected_features, selected_indices = binary_waterwheel_plant_algorithm(
        X, y, 
        n_plants=n_plants_input, 
        max_iter=max_iter_input
    )
    
    X_final = X[:, selected_indices]
    print("\nShape of original data:", X.shape)
    print("Shape of data after feature selection:", X_final.shape)
