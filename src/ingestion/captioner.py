from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(pil_image) -> str:

    input = processor(images=pil_image, return_tensors="pt")

    output_ids = model.generate(**input)

    caption = processor.decode(output_ids[0], skip_special_tokens=True)

    return caption

# processor converts PIL image → tensor done
# model generates token ids done
# processor decodes token ids → string done