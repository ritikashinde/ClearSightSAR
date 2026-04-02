# ☁️ Novel DMDiff: Latent Diffusion for SAR-Guided Cloud Removal

An implementation of a custom Latent Diffusion Model (LDM) designed to reconstruct cloud-free optical satellite imagery by conditioning on Synthetic Aperture Radar (SAR) data.

## 🚀 Project Overview
Optical satellite imagery is frequently obstructed by thick cloud cover, rendering it useless for geospatial analysis. This project solves that by utilizing a Vision-Language/Latent Diffusion architecture. By pairing cloudy optical images with cloud-penetrating SAR data, the model learns the underlying geological structures and successfully "inpaints" the missing landscape.

## 🧠 Architecture
* **Variational Autoencoder (VAE):** A custom CNN-based autoencoder trained to compress 256x256 high-resolution satellite patches into dense 32x32 latent representations, drastically reducing compute requirements.
* **Conditional UNet:** The core diffusion model that denoises the latents. It is conditioned on both the cloudy optical latent and the SAR structural latent.
* **Denoising Scheduler:** Utilizes a DDPM scheduler for the reverse diffusion process.

## 📊 Results & Metrics
The model was trained and evaluated locally under strict compute constraints, successfully proving the viability of the architecture.

* **PSNR:** 16.0812
* **SSIM:** 0.2431
* **LPIPS:** 0.8158

### Visual Reconstruction
*(Add your screenshot here! Just drag and drop the `final_generation_result.png` right into GitHub's web editor when you edit the README, and it will auto-generate the markdown link for you!)*
![Cloud Removal Results](link-to-your-image-goes-here)

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, HuggingFace Diffusers
* **Computer Vision:** OpenCV, LPIPS Perceptual Loss
* **Data Pipeline:** (Upcoming: Docker & Apache Kafka for real-time streaming inference)

## ⚙️ Local Setup
```bash
# Clone the repository
git clone [https://github.com/ritikashinde/YOUR-REPO-NAME.git](https://github.com/ritikashinde/YOUR-REPO-NAME.git)

# Install dependencies
pip install -r requirements.txt

# Run inference
python scripts/04_generate_image.py
