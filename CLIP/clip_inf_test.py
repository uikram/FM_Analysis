import os
import torch
import clip
from PIL import Image

# --- 1. Set up your specific GPU (from your code) ---
# This line MUST come BEFORE any torch.cuda calls
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# --- 2. Check the device ---
if torch.cuda.is_available():
    print(f"PyTorch sees GPU: {torch.cuda.get_device_name(0)}") # It should see GPU 2 as device 0
    print(f"Current device index: {torch.cuda.current_device()}") # Should be 0
    # Any tensors moved to 'cuda' will now go to GPU 2 by default
    device = torch.device("cuda") 
else:
    print("CUDA is not available. Check drivers or PyTorch installation.")
    device = torch.device("cpu")

print(f"--- Using device: {device} ---")

# --- 3. Load the CLIP Model ---
# The model will be loaded onto the device specified above (GPU 2)
try:
    model, preprocess = clip.load("ViT-L/14", device=device)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your internet connection or the model name.")
    exit()

# --- 4. Prepare your Image ---
# This assumes "cat.jpg" is in the same folder as this script
try:
    image = Image.open("dog.jpg")
except FileNotFoundError:
    print("\nError: 'cat.jpg' not found.")
    print("Please make sure the image is in the same folder as this .py file.")
    exit()

# Apply the model's required transformations
image_input = preprocess(image).unsqueeze(0).to(device)

# --- 5. Prepare your Text Descriptions ---
text_descriptions = [
    "a photo of a cat", 
    "a photo of a dog", 
    "a photo of a tiger", 
    "a photo of a car",
    "a picture of a person"
]
# "Tokenize" the text descriptions
text_inputs = clip.tokenize(text_descriptions).to(device)

# --- 6. Run Inference ---
# Calculate features and similarity
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    
    logits_per_image, logits_per_text = model(image_input, text_inputs)
    # Convert similarity scores to probabilities
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# --- 7. Print the Results ---
print("\nImage-Text Similarity Probabilities:")
print("---------------------------------")
for i, description in enumerate(text_descriptions):
    print(f"{description}: {probs[0][i]*100:.2f}%")
