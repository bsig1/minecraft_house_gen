# Architecture V2

Blocks are represented as vectors in a latent space where similar blocks are close together in latent space by cosine distance.

Data from https://www.kaggle.com/datasets/shauncomino/minecraft-builds-dataset/data

---

## Noising

The noising process follows the standard diffusion framework. Over a fixed schedule, Gaussian noise is added to each block’s latent representation, gradually destroying semantic and structural information until the latent tensor is approximately pure noise.

A consequence of this design is that, once noise has been added, a latent vector generally no longer corresponds exactly to any valid block embedding in the vocabulary. This is not a problem during diffusion itself, since intermediate states are allowed to exist in continuous latent space rather than correspond to discrete blocks.

---

## Denoising

The denoising process learns to reverse the forward noising procedure by gradually transforming a noisy latent tensor back into a coherent structure.

At each timestep, the model takes as input:
- a noisy latent tensor
- a timestep embedding

and predicts the noise that was added to the clean latent representation.

This denoising model is trained primarily with the standard diffusion objective: given a real build, partially noise it to a randomly chosen timestep, and train the model to predict the added noise. 

---

### Final Decoding

After the final denoising step produces a clean latent tensor, each voxel is decoded into a block.

For each latent vector:
- compute distance to all block embeddings
- convert distances to probabilities via softmax
- choose a block via argmax or sampling
  - argmax for stable, consistent builds
  - sampling for more diverse, natural variation

---

### Model Architecture

The denoising model is a 3D convolutional neural network operating over the latent voxel grid.

Key properties:
- captures spatial structure through convolution
- uses residual blocks for stable training
- incorporates timestep embeddings so the model knows how much noise is present

The architecture is designed to capture multiple scales:
- early layers focus on local patterns
- deeper layers capture larger spatial relationships

---

### Loss Guided Training

The standard diffusion loss teaches the model to reconstruct the original latent representation from a noisy input. However, reconstruction accuracy alone does not guarantee that the reconstructed build is architecturally plausible.

To address this, the model’s predicted clean latent representation is decoded into block space, and the three structural losses are applied to that reconstruction:
- palette loss checks whether the material distribution is realistic
- local structure loss checks whether small motifs resemble real building components
- global structure loss checks whether those motifs are assembled into a coherent whole

These losses act as structural regularizers. They do not replace the denoising objective. Instead, they guide the model toward reconstructions that are not only close to the original data, but also explicitly reasonable as houses.

The total training objective is therefore a weighted combination of:
- denoising accuracy
- palette realism
- local motif realism
- global structural realism

Conceptually, the training step looks like this:
1. take a real build
2. embed it into latent space
3. add Gaussian noise at a randomly chosen timestep
4. predict the added noise with the denoising network
5. reconstruct the model’s estimate of the clean latent vector
6. decode that estimate into block space
7. apply palette, local, and global losses to the reconstruction
8. backpropagate all losses together

In this way, the structural losses influence training by pushing the model to predict noise in a way that leads to better reconstructions.


---

## Loss Function

The loss function is composed of three components.

### 1. Reasonable Block Palette

Valid houses are materially multimodal, so the dataset should not be modeled as having one average palette. Instead, builds are grouped into several palette families, such as wood cabins, sandstone structures, or quartz mansions.

Each build is embedded as a weighted combination of its block embeddings and assigned to a palette cluster. For a generated build, its palette embedding is compared to the nearest cluster centroid using cosine distance, scaled by that cluster’s empirical spread.

If the generated build lies several cluster-standard-deviations away from every centroid, then its material distribution is treated as implausible.

---

### 2. Recognizable Local Structures

A lightly pooled CNN is used to detect common local building motifs such as windows, doors, roof edges, corners, pillars, and wall patterns.

This part of the loss penalizes local arrangements of blocks that do not resemble motifs commonly found in the training data. Because convolution is translationally equivariant and pooling adds some translational tolerance, this detector can recognize a window-like pattern regardless of its precise location.

However, local plausibility alone is not sufficient. A valid motif in an invalid place, such as a floating window detached from the rest of the structure, may still score well locally.

---

### 3. Global Structure

A more strongly pooled CNN evaluates the arrangement of large-scale components across the entire build.

Its role is to ensure that local motifs occur in coherent spatial configurations. For example:
- roofs should appear above walls
- doors should connect to reachable entrances
- windows should be embedded within walls rather than floating in space
- the overall massing should resemble a plausible building

---
