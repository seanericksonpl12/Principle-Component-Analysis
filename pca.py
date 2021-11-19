from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # TODO: add your code here
    x = np.load(filename)
    
    return (x - np.mean(x, axis=0))

def get_covariance(dataset):
    # TODO: add your code here
    y = (1/(len(dataset) - 1))
    covariance = y * (np.dot(np.transpose(dataset), dataset))

    return covariance

def get_eig(S, m):
    # TODO: add your code here
    eigvals, eigvect = eigh(S, subset_by_index=[len(S) - m, len(S)-1])

    # Sort from low to high, reverse order, and put in diagonal matrix
    Lambda = np.diag(np.flip(np.sort(eigvals)))
    # flip vectors on y axis to correspond with values
    U = np.flip(eigvect, 1)

    return Lambda, U

def get_eig_perc(S, perc):
    # TODO: add your code here
    # calculate minimum cut-off for eigvals
    eigmin = (np.sum(eigh(S, eigvals_only=True))) * perc

    # Same as get_eig, only filtering by value instead of index
    eigvals, eigvect = eigh(S, subset_by_value=[eigmin, np.inf])
    Lambda = np.diag(np.flip(np.sort(eigvals)))
    U = np.flip(eigvect, 1)

    return Lambda, U

def project_image(img, U):
    # TODO: add your code here
    # proj stores each value of the projection
    proj = []

    # Find the row of alphas
    alpha = np.dot(np.transpose(U), img)

    # Dot product of alpha row with each eigenvector
    m = 0
    while m < len(U):
        proj.append(np.dot(alpha, U[m]))
        m += 1
    
    # Return projection as numpy array
    return np.array(proj)


def display_image(orig, proj):
    # TODO: add your code here
    # reshape to 32x32 matrix
    reshaped_orig = orig.reshape(32, 32, order='F')
    reshaped_proj = proj.reshape(32, 32, order='F')

    # create subplot and add images, color bar and titles
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].set_title('Original')
    axs[1].set_title('Projection')
    img1 = axs[0].imshow(reshaped_orig, aspect='equal')
    img2 = axs[1].imshow(reshaped_proj, aspect='equal')
    fig.colorbar(img1, ax=axs[0])
    fig.colorbar(img2, ax=axs[1])
    
    plt.show()


    
    
