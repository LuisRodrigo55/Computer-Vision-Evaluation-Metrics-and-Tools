import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import warnings

# Lets avoid having warnings from the pre-trained model initializations
warnings.filterwarnings("ignore")

def GenerateMockImageBatches(batch_size=10, channels=3, size=299):
    #This function generates mock batches of images formatted for pre-trained networks.
    #The images must be integers in [0, 255]

    #This are the real images
    real_images = torch.randint(0, 255, (batch_size, channels, size, size), dtype=torch.uint8)
    
    #This are the generated images from the model we want to evaluate
    #We add some noise to make them different from the real images
    fake_images = torch.clamp(real_images + torch.randint(-20, 20, (batch_size, channels, size, size)), 0, 255).to(torch.uint8)
    
    return real_images, fake_images

def GenerativeEvaluation():
    print(" Generative Models & Editing Metrics ")
    
    try:
        #Fréchet Inception Distance (FID)
        print("FID Metric")
        
        #feature=64 uses the first max pooling layer of Inception v3 for speed in this mock.
        fid = FrechetInceptionDistance(feature=64, normalize=False)
        
        #Get mock data
        real_images, fake_images = GenerateMockImageBatches(batch_size=5, size=64)
        
        #Update metric with real and fake batches
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)
        
        #Compute final FID score
        fid_score = fid.compute()
        print(f"Mock FID Score: {fid_score.item():.4f}")
        print("(Lower is better. A score of 0.0 means the distributions are identical.)")
        
    except Exception as e:
        print(f"Error executing FID calculation. Ensure torchmetrics is fully installed: {e}")

    print("\n Perceptual Metrics ")
    #To calculate LPIPS or CLIP Score in your projects, you will need to install:")
    #LPIPS: `pip install lpips` 
    #CLIP: `pip install transformers`

if __name__ == "__main__":
    GenerativeEvaluation()
