If github in unable to render a Jupyter notebook, copy the link of the notebook and enter into the nbviewer: https://nbviewer.jupyter.org/

# Support Vector Machine (SVM) Classification: Linearly Separable Data

SVM is a discriminative learning modek that makes an **assumption on the form of the discriminant (decision boundary)** between the classes.

In a binary classification problem, the form of the SVM discriminant function is modeled by the **widest possible boundary** between the classes. That's why SVM is known as **large margin classifier**. 

The goal of the large margin SVM classifier is to maximize the margin between two classes. However it has to respect a **constraint**: 
    
    -- The margin should be maximized while making sure that data are correctly classified (i.e., data belonging to two classes are off the margin).

Hence the SVM classification problem can be modeled as a constrained maximization problem.

Depending on the nature of the data, the SVM constrained maximization algorithm varies.

To understand different SVM algorithms and approaches, we will consider several cases in this notebook series on SVM.

- Linearly Separable Data
        
        -- No Outlier
        -- Outlier

- Linearly Non-Separable Data
        
        -- Feature Augmentation
        -- Kernelized SVM: Polynomial Kernel
        -- Kernelized SVM: Gaussian Radial Basis Function (RBF) Kernel



## Index for the Notebook Series on SVM Classifier

There are 5 notebooks on SVM based classifiers.

1. Support Vector Machine-1-Linearly Separable Data
        
        -- Hard margin & soft margin classifier using the LinearSVC model

2. Support Vector Machine-2-Nonlinear Data
        
        -- Polynomial models with LinearSVC and Kernelized SVM (Polynomial & Gaussian RBF kernel)

3. Support Vector Machine-3-Gaussian RBF Kernel
        
        -- In depth investigation of Gaussian RBF Kernel (how to fine tune the hyperparameters)
        
4. Support Vector Machine-4-Multiclass Classification
        
        -- Multiclass classification using the SVC class that implements the One-versus-One (OvO) technique

Finally, we will apply SVM on **two application scenarios**. We will see that these two applications require two very different SVM algorithms (linear and complex models). We will conduct in dept investigations on these two models in the context of these two applications.

4. Application 1 - Image Classification (Gaussian RBF model performs well & why)

5. Application 2 - Text Classification (LinearSVC performs well & why)



## Mathematical Foundation of SVM

There are at least two very different ways to find the maximum margin decision boundary.

- Modeling the max margin problem as a constrained optimization problem and solove it using Quadratic Programming (QP) solver

- Modeling the max margin problem as an unconstrained optimization problem and solve it using Gradient Descent/coordinate descent 


### Constrained Optimization Problem

We can model the max margin problem as a constrained optimization problem in two ways.

- Primal Problem (computationally expensive for large feature dimension)

- Dual Problem


### Primal Problem

The SVM finds the max margin decision boundary by solving the following constrained optimization problem.

$min_{\vec{w}, b} \frac{1}{2}\vec{w}^T.\vec{w} + C\sum_{i=1}^{N} \xi_i$

Subject to the following constraints:

$y_i(\vec{w}^T.\vec{x}_i + b) \geq 1 - \xi_i$   $\forall i$

$\xi_i \geq 0$



Here:
- $\xi$: slack variable that controls margin violation. \#($\xi > 0$) = the number of non-separable points (measure of error/misclassification).

- C: regularization/penalty. Controls the trade-off between margin maximization and error minimization.

This convex optimization problem is known as the **primal problem** and its complexity depends on feature dimension.

We can use a Quadratic Programming (QP) solver to find optimal $\vec{w}$ and b for the primal problem:
https://cvxopt.org/


     
### Dual Problem 

Due to the computational complexity of the primal optimization (minimization) problem, we transform it into a form such that its complexity no more depends on the feature dimension, instead depends on the size of the data. This new form is known as the dual form and we solve the **dual optimization (maximization) problem**.

$max_{\alpha_1, ..\alpha_N} \sum_{i=1}^{N}\alpha_i - \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \vec{x_i}^T.\vec{x_j} $

Subject to the constraints:

$\sum_{i=1}^{N}\alpha_i y_i = 0$

$\alpha_i \geq 0 $

Here $\alpha$ is the Lagrange multiplier.

The complexity of the dual problem depends on the size of the training data (N), not on the feature dimension. Thus, for high-dimensional data, solving the dual problem is much more efficient than solving the primal problem.



## Unconstrained Optimization Problem for Gradient Descent

We can implement the gradient descent (GD) or stochastic gradient descent (SGD) or coordinate descent (CD) based approach to find optimal $\vec{w}$ and $b$ for the SVM classsifier. The SGD aproach is useful for **online learning**.

To apply these iterative optimazation approaches for the SVM, we define the cost function as follows.


$min_{\vec{w},b}\frac{1}{2}\vec{w}^T.\vec{w} + C\sum_{i=1}^{N}h(y_i(\vec{w}^T.\vec{x}_i + b)) $


Here:
- $h(z)$: the Hinge loss function: $h(z) = max(0, 1 - z)$
- C: regularization/penalty. Controls the trade-off between margin maximization and error minimization.

The Hinge loss function varies between 0 and $(1 - y_i(\vec{w}^T.\vec{x}_i + b))$. It represents the error/loss due to misclassification.

<img src="https://cse.unl.edu/~hasan/SVM1.png" width=400 height=300>

Observe that the Hinge loss or cost function of SVM is similar to the Linear Regression and Logistic Regression regularized cost function.

In case of SVM:
- The first term is the regularization/penalty term
- The second term is the loss objective function

Unlike Linear/Logistic regression, the regularization/penalty parameter (C) is with the loss function.


It's a hyperparameter that controls the trade-off between margin maximization and error minimization.
     
     - If C is too large, we have a high penalty for nonseparable points, and we may store many support vectors and overfit. 
     - If C is too small, we may find too simple solutions that underfit. 
     


## SVM using Scikit-Learn


Scikit-Learn provides four SVM models to perform classification:

- SVC (Solves the dual optimization problem. Used to implement kernelized SVM, such as polynomial kernel, Gaussian Radial Basis Function or RBF kernel)

- LinearSVC (Uses the Coordinate Descent approach. Similar to SVC with linear kernel)

- NuSVC (Nu-Support Vector Classification. Similar to SVC but uses a parameter to control the number of support vectors)

- SGDClassifier (Uses Stochastic Gradient Descent approach)

We will investigate both SVC and LinearSVC in greater detail. Also for the image classsification application we will use the SGDClasssifier.


## Scikit-Learn SVM Model Complexity


- SVC: $O(N^2d)$ ~ $O(N^3d)$

- LinearSVC: $O(Nd)$ 

N: No. of training data

d: No. of features


The LinearSVC class is based on the liblinear library, which implements an optimized algorithm for linear SVMs. It does not support the kernel trick, but it scales almost linearly with the number of training instances and the number of features ($O(Nd)$). Moreover, the LinearSVC class has more flexibility in the choice of penalties (l2 & l1) and loss functions. 


The SVC class is based on the libsvm library, which implements an algorithm that supports the kernel trick. Due to its complexity between $O(N^2d)$ ~ $O(N^3d)$, it gets dreadfully slow when N gets large (e.g., hundreds of thousands of instances). However, SVC is perfect for **complex but small or medium** training sets. It scales well with the number of features, especially with sparse features (i.e., when each instance has few nonzero features). 


## How do We Choose the Optimal Model (between SVC & LinearSVC)?

Model selection is done by hyperparameter tuning. We can choose both the algorithm (LilearSVC, SVC with varying kernels) and the optimal hyperparameters via cross-validated grid search.

However, brute-force grid search is time consuming. We should have a high-level understanding of the suitability of the algorithms based on the dataset. Then, we can fine tune the hyperparameters.

So, before doing any Machine Learning with SVM, we should address these questions.

- How do we choose the most suitable model between LinearSVC and SVC?

- If SVC is suitable, then how do we choose the optimal kernel (usually between polynomial and RBF)?



# Guideline (Rough) to Choose the Suitable Model Based on the Data

- N is very large but d is small ($N > d$): LinearSVC

- d is large relative to N ($d \geq N$): LinearSVC

- N is small to medium and d small ($N > d$): SVC with Gaussian RBF kernel


In this notebook we will classify a linearly separable dataset. Both N and d are small in this dataset. Also, it is linearly separable.

Thus, will use the **LinearSVC** model. 



## LinearSVC Class: Hyperparameter Setting:

- The "loss" hyperparameter should be set to "hinge". 

- The hyperparameter "C" controls the penalty for the error (margin violation). It should be selected via grid search. We will investigate its effect shortly.

- Finally, for better performance we should set the "dual" hyperparameter to False, unless there are more features than training instances.



### Scaling

The SVM classfication is influenced by the varying scale of the features. 

SVMs try to fit the largest possible “street” between the classes. So if the training set is not scaled, the SVM will tend to neglect small features. 


Thus, we should standardize the data before training.


## Linearly Separable Data: Two Cases

We will consider two cases.

- Data doesn't have outlier

- Data has outliers
