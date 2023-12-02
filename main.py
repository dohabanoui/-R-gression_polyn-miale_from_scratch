import numpy as np
import matplotlib.pyplot as plt


def plot_4_cases(x, y,model_list):
    couleur = ['red','green','pink','yellow']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for i in range(len(model_list)):
        row, col = divmod(i, 2)

        axes[row, col].plot(x, y, 'o',label='dataset')
        axes[row, col].plot(x, model_list[i], 'o', linestyle='None', color=couleur[i], label=f'Cas {i+1}')
        axes[row, col].set_xlabel('petal_length')
        axes[row, col].set_ylabel('sepal_width')
        axes[row, col].legend()

    plt.tight_layout()

    # Affichage de la figure
    plt.show()



def ploter(x,y,model):
    plt.plot(x,y,'o',label='dataset')
    plt.plot(x, model,'o',c='green')
    plt.xlabel('petal_length')
    plt.ylabel('sepal_width')
    plt.legend()
    plt.show()

# définir le modèle
def model(X, theta):

    return X.dot(theta)



#définir la fonction coût
def cost_function(X, y, theta): 
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)


# définir la fonction de gradient
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)


# Définir la descente de gradient
from tqdm import tqdm

# Définir la descente de gradient avec une barre de progression
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)  # Création d'un tableau de stockage pour enregistrer l'évolution du coût du modèle

    # Utilisation de tqdm pour créer une barre de progression
    for i in tqdm(range(n_iterations), desc='Descente de gradient'):
        theta = theta - learning_rate * grad(X, y, theta)  # Mise à jour du paramètre theta (formule de la descente de gradient)
        cost_history[i] = cost_function(X, y, theta)  # On enregistre la valeur du coût au tour i dans cost_history[i]

    return theta, cost_history


def cost_function_reg(X, y, theta,_lambda): 
    m = len(y)
    regularization = (_lambda/(2*m)) * np.sum(theta[1:]**2)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2) + regularization


def gradient_regularized(X, y, theta,_lambda):
    m = len(y)
    regularization_term = ((_lambda/m) *  theta)
    gradient = 1/m * X.T.dot(model(X, theta) - y) + regularization_term
    return gradient


def gradient_descent_regularized(X, y, theta, learning_rate, n_iterations, _lambda):
    cost_history = np.zeros(n_iterations)

    for i in tqdm(range(n_iterations), desc='Descente de gradient'):
        theta = theta - learning_rate * gradient_regularized(X, y, theta,_lambda)
        cost_history[i] = cost_function_reg(X, y, theta, _lambda)

    return theta, cost_history


def cost_function_regLasso(X, y, theta,_lambda): 
    m = len(y)
    regularization = (_lambda/(2*m)) * np.linalg.norm(theta[1:])
    return 1/(2*m) * np.sum((model(X, theta) - y)**2) + regularization

def gradient_regLasso(X, y, theta,_lambda):
    m = len(y)
    regularization_term = ((_lambda/m) *  np.sign(theta))
    gradient = 1/m * X.T.dot(model(X, theta) - y) + regularization_term
    return gradient


def gradient_descent_regLasso(X, y, theta, learning_rate, n_iterations, _lambda):
    cost_history = np.zeros(n_iterations)

    for i in tqdm(range(n_iterations), desc='Descente de gradient'):
        theta = theta - learning_rate * gradient_regLasso(X, y, theta,_lambda)
        cost_history[i] = cost_function_reg(X, y, theta, _lambda)

    return theta, cost_history


# Evaluation du modèle
def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v