# Linear model (LM)

## Problem Formulation

**Linear Model (LM)** is a basic **regression** algorithm. The matrix representation of LM is:

\[Y = X \beta + \epsilon, \text{ where } \epsilon \sim N(0, \sigma^2I)\]

**Note:**  In some situations, \( \epsilon \) may not follow **Normal Distribution**. The algorithm is called **General Linear Model (GLS)** in this case.

\[Y = X \beta + \epsilon, \text{ where } \epsilon \sim N(0, V) \text{ for some } V\]

- **Heteroscedasticity**: unequal error variance
- **Serial Correlation**: correlated errors

## Least Squares Estimate of Parameters of OLS

- Linear model
\[Y = X \beta + \epsilon, \text{ where } \epsilon \sim N(0, \sigma^2I)\]

- Error sum of squares
\[
\begin{aligned}
S(\beta) = \epsilon^T\epsilon &= (Y - X \beta)^T (Y - X \beta)\\
& = Y^TY -Y^TX\beta  - \beta^TX^TY + \beta^TX^TX\beta \\
& = Y^TY - 2\beta^TX^TY + \beta^TX^TX\beta
\end{aligned}
\]

- Differentiation and solve
\[
    \frac{\partial S(\beta)}{\partial \beta} = -2X^TY + X^TX\beta + (\beta^TX^TX)^T = -2X^TY + 2X^TX\beta = 0
\]

    **Matrix Differentiation Trick**: 
    1. suppose the shapes of matrices are: 
    \[Y \in \mathbb{R}^{n \times 1}; \ X \in \mathbb{R}^{n \times (1+p)}; \ \beta \in \mathbb{R}^{(1+p) \times 1} \]
    2. suppose the derivative \( \frac{\partial S(\beta)}{\partial \beta} \in \mathbb{R}^{(1+p) \times 1}\), then analogize to scalar calculus:
    \[X^TY \in \mathbb{R}^{(1+p) \times 1}; \ X^TX\beta \in \mathbb{R}^{(1+p) \times 1}; \ \beta^TX^TX \in \mathbb{R}^{1 \times (1+p)} \]
    3. reshape inconsistent part:
    \[
        \text{reshape } \beta^TX^TX \rarr (\beta^TX^TX )^T=X^TX\beta \\
        \frac{\partial S(\beta)}{\partial \beta} = -2X^TY + 2X^TX\beta
    \]

- Ordinary Least Squares Extimator (OLS)
\[
    \hat{\beta} = (X^TX)^{-1}X^TY
\]

## Statistical Inference

### Mean and Variance of \( \hat{\beta} \)

- Problem formulation and estimator:

\[Y = X \beta + \epsilon, \text{ where } \epsilon \sim N(0, \sigma^2I)\]

\[\hat{\beta} = (X^TX)^{-1}X^TY = LY, \text{ where } L = (X^TX)^{-1}X^T\]

- **The uncertainty (variance) comes from \( \epsilon \) by assumption**:

\[
\begin{aligned}
    \mathbb{E}(\hat{\beta}) &= \mathbb{E}(LY) = L\mathbb{E}(X\beta + \epsilon) = LX\beta \\
    &= (X^TX)^{-1}X^TX\beta = \beta
\end{aligned}
\]

\[
\begin{aligned}
    \text{Var}(\hat{\beta}) &= \text{Var}(LY) = L\text{Var}(Y)L^T = L\text{Var}(\epsilon)L^T \\
    &= L(\sigma^2I)L^T = \sigma^2 (X^TX)^{-1}X^TX(X^TX)^{-1} = \sigma^2 (X^TX)^{-1} \\
    & \text{where} \ ((X^TX)^{-1})^T = ((X^TX)^T)^{-1} = (X^TX)^{-1}
\end{aligned}
\]

### Inference of \( \hat{\beta_k} \)
- Since we suppose \( \epsilon \) follows **Normal Distribution**, so does \( \hat{\beta_k} \) and we have **Least squares estimate**:

\[
\hat{\beta} = LY \sim N(\beta, \sigma^2(X^TX)^{-1}) \\
\hat{\beta_k} \sim N(\beta_k,c_{kk}\sigma^2), \text{ where } c_{kk} \text{ is the } k^{th} \text{ diagonal element of } (X^TX)^{-1}
\]

- Under the **Null Hypothesis** that \( H_0: \beta_k = 0 \), which means the \( k^{th} \) has not predictive power, we have:

\[ T = \frac{\hat{\beta_k}}{\hat{\sigma} \sqrt{c_{kk}}}, \text{ since } \sigma \text{ is unknown} \]

- Now estimate \( \hat{\hat{\sigma}} \) by **Residual Sum of Squares (RSS)**:
    1. the prediction is given by:
    \[
    \begin{aligned}
        \hat{Y} = X\hat{\beta} &= X(X^TX)^{-1}X^TY = PY \\
        & \text{ where } P = X(X^TX)^{-1}X^T 
    \end{aligned}
    \]
    \( P \), the projection matrix, is **symmetric** and **idempotent**:
    \[
    P^T = (X(X^TX)^{-1}X^T)^T = X((X^TX)^T)^{-1}X^T = X(X^TX)^{-1}X^T = P \\
    P^2 = PP = X\sout{(X^TX)^{-1}X^TX}(X^TX)^{-1}X^T= X(X^TX)^{-1}X^T
    \]
    
    2. compute the **RSS** via:
    \[
    \begin{aligned}
        e &= Y - \hat{Y} = (I-P)Y\\
        RSS &= e^Te = (Y-\hat{Y})^T(Y-\hat{Y}) = ((I-P)Y)^T((I-P)Y) \\
        &= Y^T(I-P)^T(I-P)Y = Y^T(I-P-P^T+P^TP)Y \\
        &= Y^T(I-P)Y \\
        RSS&= (X\beta + \epsilon)^T (I-P) (X\beta + \epsilon)\\
        &= \beta^T X^T (I-P) X \beta + \beta^T X^T (I-P) \epsilon + \epsilon^T (I-P) X\beta + \epsilon^T (I-P) \epsilon \\
        &= \epsilon^T (I-P) \epsilon \\
        &\text{since } (I-P) X = X - PX = X-X\sout{(X^TX)^{-1}X^T X} = 0 \\ 
        &\text{and } X^T (I-P) = ((I-P)X)^T = 0
    \end{aligned}
    \]

    3. now the **trace-variance formula** need to be proved for later derivation:
        1.  firstly I need to proce a proposition:
        \[
        \text{Proposition: } tr(ABC) = tr(BCA) = tr(CAB) \\
        \text{Proof: suppose the shapes are: } A \in \mathbb{R}^{m \times n}; B \in \mathbb{R}^{n \times p}; C \in \mathbb{R}^{p \times m}\\
        D = ABC; \ E = BCA; \ F = CAB, \text{ then} \\
        tr(D) = \sum_{i=1}^m D_{ii} = \sum_{i=1}^m (\sum_{j=1}^n A_{ij}(\sum_{k=1}^p B_{jk}C_{ki})) = \sum_{i=1}^m \sum_{j=1}^n \sum_{k=1}^p A_{ij}B_{jk}C_{ki} \\
        tr(E) = \sum_{j=1}^n E_{ii} = \sum_{j=1}^n (\sum_{k=1}^p B_{jk}(\sum_{i=1}^m C_{ki}A_{ij})) = \sum_{j=1}^n \sum_{k=1}^p \sum_{i=1}^m B_{ij}C_{jk}A_{ki} \\
        tr(F) = \sum_{k=1}^p F_{kk} = \sum_{k=1}^p (\sum_{i=1}^m C_{ki}(\sum_{j=1}^n A_{ij}B_{jk})) = \sum_{k=1}^p \sum_{i=1}^m \sum_{j=1}^n C_{ki}A_{ij}B_{jk} \\
        \Rarr tr(D) = tr(E) = tr(F)
        \]
        2.  secondly prove the **trace-variance formula**:
        \[
            \text{Trace-Variance: assume } A \text{ is a quadratic form, then } A^T = A \text{ we have} \\ 
            \mathbb{E}(y^TAy) = tr(A\Sigma) + \mu^TA\mu, \text{ where } \mu = \mathbb{E}(y), \ \Sigma = \text{Cov}(y) \\
            \text{Proof: let } z = y - \mu, \text{ then } \mathbb{E}(z) = 0, \text{ and since } y = z + \mu, \text{ Cov}(z) = \Sigma \\
            y^TAy = (\mu + z)^TA(\mu + z) = \mu^TA\mu + 2\mu^TAz + z^TAz \text{ since } A^T = A \\
            \begin{split}
                \mathbb{E}(y^TAy) &= \mathbb{E}(\mu^TA\mu) + 2\mathbb{E}(\mu^TAz) + \mathbb{E}(z^TAz) \\
                &= \mu^TA\mu + \sout{2\mu^TA\mathbb{E}(z)} + tr(z^TAz) \\
                &= \mu^TA\mu + tr(Azz^T) \\
                &= \mu^TA\mu + tr(A\Sigma)
            \end{split}\\
            \text{since } \mu \ \&\  A \text{ are not random, we have } \mathbb{E}(\mu^TAz) = \mu^TA\mathbb{E}(z); \\
            \text{for trace, we have: } tr(ABC) = tr(BCA) = tr(CAB); \\
            zz^T = \Sigma \Rarr tr(z^TAz) = tr(Azz^T) = tr(A\Sigma)
        \]
    4. the expectation of **RSS** is:
    \[
        \begin{aligned}
            \mathbb{E}(RSS) &= \mathbb{E}(Y^T(I-P)Y) = \mathbb{E}(Y)^T(I-P)\mathbb{E}(Y) + tr((I-P)\text{Var}(Y)) \\
            &= \beta^TX^T(I-P)X\beta + tr((I-P)\sigma^2I) \\
            &= \beta^TX^T(X-PX)\beta + \sigma^2(tr(I) - tr(P)) \\
            &= \beta^TX^T(X-X(X^TX)^{-1}X^TX)\beta + \sigma^2(tr(I) - tr(P)) \\
            &= \sigma^2(n - tr(X(X^TX)^{-1}X^T)) \\
            &= \sigma^2(n - tr((X^TX)^{-1}X^TX)) \\
            &= \sigma^2(n-p-1) \\
            & \text{where the degree of freedom is } n-p-1 \\
            \mathbb{E}(RSS) &= \mathbb{E}(\epsilon^T(I-P)\epsilon) = tr(\epsilon^T(I-P)\epsilon) = tr((I-P)\epsilon\epsilon^T) \\
            &= tr((I-P)\sigma^2I) = \sigma^2tr(I-P) = \sigma^2(tr(I) - tr(p)) \\ 
            &=\sigma^2(n-p-1)
        \end{aligned}
    \]
    5. estimate of error variance \( \hat{\sigma} \) is: 
    \[
        \hat{\sigma}^2 = \frac{RSS}{n-p-1} \Rarr \hat{\sigma} = \sqrt{\frac{RSS}{n-p-1}}
    \]
    6. test whether \( \beta_k \) is significant by **Two sided T test**:
    \[
        H_0: \beta_k = 0 \\
        T = \frac{\hat{\beta_k}}{\hat{\sigma}\sqrt{c_{kk}}}  
    \]

### Inference of *Reduced Model* against **Full Model**

The precentage of **unexplained variance** is important measurement to determine whether a model fits well or not. High precentage of unexplained variance means underfitting while low precentage may leads to overfitting.

- Suppose the full model has \( p \) parameters \( \beta_1, \cdots, \beta_k \) and 1 interception \( \beta_0 \) and the reduced model remove \( p-q \) parameters from \( \beta_1, \cdots, \beta_k \):
    \[
        \text{Full model: } Y = X\beta_P + \epsilon, \text{ where } \beta_P = [\beta_0, \beta_1, \cdots, \beta_p]^T \\
        \text{Reduced model: } Y = X\beta_Q + \epsilon \ (q < p),  \text{ where } \beta_Q = [\beta_0, \beta_1, \cdots, \beta_q]^T
    \]

- Fit the full model and the reduced model, then obtain their **RSS** by:
    \[
        RSS = \epsilon^T\epsilon = (Y-\hat{Y})^T(Y-\hat{Y}) = Y^T(I-P)Y
    \]

- The null hypothesis is that the reduced model is good enough. That is to say, the reduced parameters have not significant influence on \( Y \). Consequently, the alternative hypothesis should be **at least one** reduced parameter has significant influence on \( Y \).
    \[
        H_0: \beta_{q+1} = \beta_{q+2} = \cdots = \beta_{p} = 0
    \]

- **F test** is applied to check whether there is significant increase of the unexplained variance precentage using the full model.
    \[
        \begin{aligned}
            F &= \frac{(RSS_{reduced} - RSS_{full}) / (df_{reduced} - df_{full}) }{ RSS_{full} / df_{full} } \\
            &= \frac{(RSS_{reduced} - RSS_{full}) / (p - q) }{ RSS_{full} / (n - p -1) } \\
            &\sim F(p-q, n-p-1)
        \end{aligned}
    \]
    If \( F \) is small, it means that including more parameters does not significantly reduce the predicted error, supports \( H_0 \), the reduced model is better. Otherwise, the full model is better.

- Now let's prove why we use **\( F \) test** here:
    1. the \( F \)-distribution is the ratio of two independent chi-squared distributions \( \chi^2 \) with degrees of freedom \( d_1 \) and \( d_2 \), respectively. The \( F \) test is used to compare whether the variances of two populations are equal.
        \[
            F = \frac{\chi^2_{d_1} / d_1}{\chi^2_{d_2} / d_2}; \ Q = \sum_{i=1}^k Z_i^2 \sim \chi^2(k), \text{ where } Z_i \sim N(0,1)
        \]
        In the model comparison context, **\( F \) test** can be used to inference the difference of unexplained variance among the two models.
    
    2. now we need to prove how **RSS** can be used to construct **F**:
        \[
            RSS = \epsilon^T(I-P)\epsilon, \text{ where } P = X(X^TX)^{-1}X^T \text{ is real symmetric} \\
            (I-P) \text{ is also real symmetric} \Rarr \text{Spectral Decomposition}\\
            \Rarr I-P = U \Lambda U^T, \text{ where } U \text{ is orthogonal: } U^TU = I \\
            \text{for } Z = \frac{\epsilon}{\sigma}  \sim N(0,I) \text{, we have } Z^T(I-P)Z = Z^T U \Lambda U^T Z \\
            \text{let } W = U^TZ, \text{ then } \mathbb{E}(W) = \mathbb{E}(U^TZ) = U^T \mathbb{E}(Z) = 0 \\
            \text{Var}(W) = \text{Var}(U^TZ) = U^T \text{Var}(Z) U = U^T U I = I \\
            \text{hence } W \sim N(0,I) \Rarr W^T \Lambda W = \sum_{i=1}^{n-p-1} w_i^2 \sim \chi^2(n-p-1) \\
            \text{and } \frac{RSS}{\sigma^2} = \frac{\epsilon}{\sigma}^T(I-P)\frac{\epsilon}{\sigma} = Z^T(I-P)Z \sim \chi^2(n-p-1)
        \] 
        [Spectral Decomposition](../spe_decom/spe_decom.md)
    3. difference of \( \chi^2 \) distribution is still \( \chi^2 \) distribution:
        \[
            Q = \sum_{i=1}^k Z_i^2 \sim \chi^2(k); \ V = \sum_{i=1}^g Z_i^2 \sim \chi^2(g), \ (k > g) \\
            Q - K = \sum_{i=1}^k Z_i^2 - \sum_{i=1}^g Z_i^2 = \sum_{i=1}^{k-g} Z_i^2 \sim \chi^2(k-g)
        \]
    4. test whether redundant parameters:
        \[
            \begin{aligned}
                H_0&: \beta_{q+1} = \beta_{q+2} = \cdots = \beta_{p} = 0 \\
                F &= \frac{(RSS_{reduced} - RSS_{full}) / (df_{reduced} - df_{full}) }{ RSS_{full} / df_{full} } \\
                &= \frac{(RSS_{reduced} - RSS_{full}) / (p - q) }{ RSS_{full} / (n - p -1) } \\
                &\sim F(p-q, n-p-1)
            \end{aligned}
        \]

### Goodness of Fit for Sigle Model

For one single linear model, we may use **\( R^2 \)** (precentage of explained variance) and **adjusted \( R^2 \)** (also take into consideration of parameter size) to evaluate the **variation** explained by the linear relationship.

- **\( R^2 \) (coefficient of determination)**: \( R^2 = 1 - \frac{RSS}{SST} \), where \( SST = \sum_{i=1}^n (y_i - \bar{y})^2 \)
- **\( adj-R^2 \) (coefficient of determination)** \( adj-R^2 = 1-\frac{n-1}{n-p-1} (1-R^2) \)
  
The limitation of the ordinary \( R^2 \) is that even if irrelevant variables are added, \( R^2 \) will artificially increase because the model can always overfit the noise. The \( adj-R^2 \) scales the precentage of unexplained variance by a factor \( \frac{n-1}{n-p-1} \), where \( n-1 \) is the degree of freedom of **interception-only model** and \( n-p-1 \) is the df of the current model. Basically, the more parameters, the heavier the penalty is.

# General Linear Mode (GLM)

## Problem Formulation

In **GLM**, \( \epsilon \) follows **Normal Distribution**. The algorithm is called **General Linear Model (GLS)** in this case.

\[Y = X \beta + \epsilon, \text{ where } \epsilon \sim N(0, V) \text{ for some } V \Rarr Y \sim N(X\beta, V)\]

- **Heteroscedasticity**: unequal error variance
- **Serial Correlation**: correlated errors

The **likelihood function** of GLM is:
\[L(\beta | X, Y) = (2\pi)^{-\frac{n}{2}} |V|^{-\frac{1}{2}} e^{-\frac{1}{2} (Y-X\beta)^T V^{-1} (Y-X\beta)} \]

## TO BE CONTINUED

# Empirical Model Diagnostics Method

## Q-Q Plot

The QQ plot is a general-purpose distribution testing tool suitable for **any model** that **requires validation of data distribution assumptions**. The x-axis represents **the quantiles of the theoretical distribution** (such as the normal distribution) as a reference benchmark. The y-axis represents **the quantiles of the actual data**, reflecting the true distribution. The QQ plot assesses the consistency between the two to determine whether the data conforms to the theoretical distribution.

## Residual Plot

The plot of **residuals vs fitted values** is useful for checking assumptions of **linearity and homoscedasticity**. To assesse the linearity assumption, we need to ensure all the residuals are not far away from 0 (suppose you have already fiited a linear model). To check the homoscedasticity, just examine the trend and shape of the residuals