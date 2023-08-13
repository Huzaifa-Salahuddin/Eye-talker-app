from flask import Flask, render_template, request
from gtts import gTTS
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = Flask(__name__)


app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():

    global model, resnet, vocab, inv_vocab

    img = request.files['file1']

    img.save('static/file.jpg')

    print("IMAGE SAVED")

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    
    image_paths = [img]
    images = []
    for image_path in image_paths:
      i_image = Image.open(image_path)
      if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
      images.append(i_image)
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]


    pre = ' '.join(preds)
    pred_2=str(pre)
   

    mytext = str(preds)
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("sound.mp3")
    # Playing the converted file
    cc=os.system("sound.mp3")
    return render_template('after.html', data=pred_2.capitalize(), sound=cc)

if __name__ == "__main__":
    app.run(debug=True)