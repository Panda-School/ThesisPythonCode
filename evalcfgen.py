import sys, argparse, torch, json
from io import BytesIO
import requests

from huggingface_hub import login
from diffusers import StableDiffusion3Img2ImgPipeline
from PIL import Image
from tqdm import tqdm
from loguru import logger
from fidFolder import compute_fid_between_folders

def get_clip_score(genImage, prompt, clip, device="cpu"):
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


def edit_away(
    image: Image.Image,
    caption: str,
    avoid_sentence: str,
    sd_pipe: StableDiffusion3Img2ImgPipeline,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50
) -> Image.Image:
    """
    Edit `image` to preserve `caption` but move it away from `avoid_sentence`.
    Uses `avoid_sentence` as negative prompt.
    """
    result = sd_pipe(
        prompt=caption,
        negative_prompt=avoid_sentence,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )
    return result.images[0]

def get_image(caption_id: str) -> Image.Image:
    """
    Load an image given its
    caption_id from a URL.
    """
    response = requests.get("https://hazeveld.org/snli-ve/images/"+ caption_id)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def main(argv):
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--amount", help='Amount of images to process, choose "None" if all images should be processed', type=str, default="None")
    parser.add_argument("--dataset", help="Dataset jsonl location", type=str, required=True)
    parser.add_argument("--token", help="Hugginface Token", type=str, required=True)
    parser.add_argument("--cuda", help="Use cuda or not", action="store_true")
    parser.add_argument("--verbose", help="Log to terminal or not", action="store_true")
    args = parser.parse_args()

    loglevel = "DEBUG" if args.verbose else "INFO"
    logger.add("eval-{time}.log", format="{name} {message}", level=loglevel, rotation="10 MB")

    #setup logging
    logger.info("Starting the programs")

    # Login to HuggingFace
    login(token=args.token)
    logger.info("Successfully logged in to HuggingFace")
    
    #clip scores
    clip_scores=dict()

    # Loading the model's pipeline
    device = "cuda" if args.cuda else "cpu"
    logger.info(f"Using device: {device}")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16).to(device)
    logger.info("Successfully loaded the model's pipeline")

    # Loading the dataset
    logger.info(f"Loading dataset from {args.dataset}")
    with open(args.dataset, "r") as f:
        data = [json.loads(line) for line in f]

    used_images = set() if args.amount != "None" else None
    am = 1

    if args.amount == "None":
        am = len(data)
    else:
        am = int(args.amount)

    # Generating images
    logger.info(f"Generating {am} images")
    for i in tqdm(range(am), desc="Generating images"):
        caption_id: str = data[i]['captionID'].split("#")[0]
        if args.amount == "None":
            used_images.add(caption_id)
        image = get_image(caption_id)
        image.save(f"originalImages/{caption_id}.png")
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        prompt = data[i]['sentence2']
        imgTowards = pipe(
            prompt=prompt, 
            image=image,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]

        imgTowards.save(f"outputTowards/{data[i]['captionID']}.png")

        imgAway = edit_away(
            image=image,
            caption=data[i]['sentence1'],
            avoid_sentence=prompt,
            sd_pipe=pipe,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=50
        )

        imgAway.save(f"outputAway/{data[i]['captionID']}.png")


    logger.info("Finished generating images")

   
if __name__ == "__main__":
   main(sys.argv)