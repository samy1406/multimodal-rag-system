import fitz
from PIL import Image
from io import BytesIO


def load_pdf(path):
    # list with all the data to return
    extracted_data = []
    # open the pdf
    doc = fitz.open(path)

    for page_num in range(len(doc)):
        # opening page in the doc
        page = doc[page_num]
        # storing text on the page
        text = page.get_text("text")

        # getting image list in the page
        image_list = page.get_images(full=True)
        
        # storing images as list
        pil_images = []

        # iterating over each image to store the the image
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"] 
            pil_img = Image.open(BytesIO(image_bytes))
            pil_images.append(pil_img)
        
        # appending the extracted data from each page
        extracted_data.append({
            "page_number": page_num + 1,
            "text": text,
            "images": pil_images
        })

    # returns list of the extracted data
    return extracted_data