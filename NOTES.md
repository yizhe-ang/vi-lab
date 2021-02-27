# TODO
- What is the best way to do KL annealing?
    - Tune the annealing schedule

- Test efficacy of multimodal fusion module
    - Transformer?

- Latent classifier evaluation
- Try learning rate finder?

# Architecture
**Approx Posterior, q(z|x)**
- Can infer from arbitrary number of inputs
    - E.g. q(z|x,y), q(z|x), q(z|y)

# Objective
Multimodal ELBO objectives require:

**Bimodal**
- z' ~ q(z|x, y)

- D(q(z'|x, y) | q(z'|x))
- D(q(z'|x, y) | q(z'|y))

- log p(x|z')
- log p(y|z')

**Unimodal**
- z_x ~ q(z|x)
- z_y ~ q(z|y)

- D(q(z_x|x) | p(z_x))
- D(q(z_y|y) | p(z_y))

- log p(x|z_x)
- log p(y|z_y)

# Experiments
- MNIST
    - For sanity check

# Evaluation
- Likelihoods
    - log p(x, y)
    - log p(x)
    - log p(y)

- Coherence
    - Joint coherence
        - z -> x, y
    - Cross coherence
        - x -> y, y -> x

# Classifier Models
- MNIST
    - 99.25% accuracy
- SVHN
    - 88.26% accuracy
    - Train a bit more?
