import sys, argparse, torch, json
from huggingface_hub import interpreter_login
from diffusers import StableDiffusion3Img2ImgPipeline
import requests
import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torch
import logging

from PIL import Image

def get_clip_score(image, prompt, clip, device="cpu"):
    # Load model and device
    assert device in ["cuda", "cpu"]
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load and preprocess image
    image = preprocess(genImage).unsqueeze(0).to(device)

    # Provide a caption
    text = clip.tokenize(prompt).to(device)

    # Get features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity (i.e., CLIP score)
    clip_score = (image_features @ text_features.T).item()
    return clip_score


def main(argv):
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing the input images", type=str)
    parser.add_argument("output", help="Path to folder to place the output images", type=str)
    parser.add_argument("--amount", help='Amount of images to process, choose "None" if all images should be processed', type=str, default="None")
    parser.add_argument("--dataset", help="Dataset jsonl location", type=str, required=True)
    parser.add_argument("--cuda", help="Use cuda or not", action="store_true")
    parser.add_argument("--clip", help="Calculate clip scores", action="store_true")
    args = parser.parse_args()

    #setup logging
    logging.basicConfig(level=logging.INFO, filename="evalcfgen.log")
    logging.info("Starting the programs")

    if(args.clip):
        import clip as cp

    # Login to HuggingFace
    interpreter_login()
    logging.info("Successfully logged in to HuggingFace")
    
    #clip scores
    clip_scores=dict()

    # Loading the model's pipeline
    device = "cuda" if args.cuda else "cpu"
    logging.info(f"Using device: {device}")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16).to(device)
    logging.info("Successfully loaded the model's pipeline")

    # Loading the dataset
    logging.info(f"Loading dataset from {args.dataset}")
    with open(args.dataset, "r") as f:
        data = [json.loads(line) for line in f]

    if args.amount == "None":
        args.amount = len(data)
    else:
        args.amount = int(args.amount)

    # Generating images
    logging.info(f"Generating {args.amount} images")
    for i in tqdm(range(args.amount), desc="Generating images"):
        caption_id: str = data[i]['captionID'].split("#")[0]
        image = Image.open(f"{args.input}/{caption_id}").convert("RGB")
        prompt = data[i]['sentence2']
        genImage = pipe(
            prompt=prompt, 
            image=image,
            num_inference_steps=30, 
            guidance_scale=7.5
        ).images[0]

        genImage.save(f"output/{data[i]['captionID']}.png")
        if(args.clip):
            clip_score = get_clip_score(genImage, prompt, cp, device=device)
            clip_scores[data[i]['captionID']] = clip_score
        #print(f"CLIP score for image {data[i]['captionID']}: {clip_score}")

    logging.info("Finished generating images")
    logging.info("Saving clip scores")
    if(args.clip):
        with open("clip_scores.json", "w") as f:
            json.dump(clip_scores, f)
   
if __name__ == "__main__":
   main(sys.argv)