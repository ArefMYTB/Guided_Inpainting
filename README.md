# Inpainting with Stable Diffusion + Reference Guidance

This project fine-tunes a **Stable Diffusion inpainting model** to perform image inpainting with additional **reference images** as guidance.  
Build up on the pretrained model [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting), and extend its UNet to condition on extra channels.

---

## Dataset

- **Source**: [MS COCO dataset](https://cocodataset.org/#home)  
- **Generation process**:
  - For each image, create a **mask** (random object/region removal).  
  - Produce:
    - **Source image** → masked version of original.  
    - **Target image** → original ground truth.  
    - **Reference image** → additional context image from the dataset.
    - **Mask image**

---

## Model Modifications

- **Base model**: `runwayml/stable-diffusion-inpainting`  
- **UNet expansion**:  
  - Extend the input channels, so the UNet can take:
    - Masked source latents  
    - Mask latents  
    - Reference latents  
- **Frozen components**:
  - **VAE** → acts only as encoder/decoder, not trained.  
  - **Text encoder** → unused (no prompt conditioning).  

This reduces training cost and stabilizes learning.

---

## Training

- **Loss**: MSE between predicted and target noise.  
- **Mixed precision**: `torch.cuda.amp` + `GradScaler`.  
- **Optimizer**: AdamW.  
- **Scheduler**: DDIMScheduler
  
