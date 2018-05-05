from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

class PlotDecisionRegions2():

    
    
    def plot_decision_regions2(X, y, classifier, test_idx=None, resolution=0.02, xlabel=None, ylabel=None):

        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict_proba(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z_0 = Z[:, 0]
        Z_1 = Z[:, 1]
        Z_2 = Z[:, 2]
        Z_0 = Z_0.reshape(xx1.shape)
        Z_1 = Z_1.reshape(xx1.shape)
        Z_2 = Z_2.reshape(xx1.shape)
        
        fig, (axL, axC, axR) = plt.subplots(ncols=3, figsize=(16,4))
        
        axL.contourf(xx1, xx2, Z_0, alpha=0.4, cmap=plt.cm.jet)
        axC.contourf(xx1, xx2, Z_1, alpha=0.4, cmap=plt.cm.jet)
        axR.contourf(xx1, xx2, Z_2, alpha=0.4, cmap=plt.cm.jet)
        
        axL.set_xlabel(xlabel)
        axC.set_xlabel(xlabel)
        axR.set_xlabel(xlabel)
        axL.set_ylabel(ylabel)
        axC.set_ylabel(ylabel)
        axR.set_ylabel(ylabel)

        for idx, cl in enumerate(np.unique(y)):
            axL.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
            axC.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
            axR.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)

        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            axL.scatter(X_test[:, 0], X_test[:, 1], facecolors='none', edgecolors='black', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')
            axC.scatter(X_test[:, 0], X_test[:, 1], facecolors='none', edgecolors='black', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')
            axR.scatter(X_test[:, 0], X_test[:, 1], facecolors='none', edgecolors='black', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')
 