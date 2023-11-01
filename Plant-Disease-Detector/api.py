from fastapi import FastAPI, Body, Request, File, UploadFile, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from fastapi import Request

from tensorflow.python.keras.backend import argmax
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

templates = Jinja2Templates(directory="script")


MODEL_POTATO = tf.keras.models.load_model("Models/Potato.h5")
POTATO_CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
POTATO_CAUSE = ["Fungus [Alternaria solani]",
                "Water Mold [Phytophthora infestans]", 
                "None"]
POTATO_DISC = ["Affects leaves, stems and tubers and can reduce yield, tuber size, storability of tubers, quality of fresh-market and processing tubers and marketability of the crop",
               "Infect potato foliage and tubers at any stage of crop development", 
               "None"]
POTATO_TREAT = ["Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate",
                "Fungicides that contain maneb, mancozeb, chlorothanolil, or fixed copper", "None"]
POTATO_PREVENTION = ["Planting potato varieties, Avoid overhead irrigation and allow for sufficient aeration between plants to allow the foliage to dry as quickly as possible",
                "Eliminating cull piles and volunteer potatoes, using proper harvesting and storage practices, and applying fungicides when necessary, Air drainage to facilitate the drying of foliage each day is important", "None"]

@app.route('/predictPotato', methods=['POST'])
def predictPotato():
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = image.resize((256,256), Image.LANCZOS)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL_APPLE.predict(img_batch)
    print(predictions)


    predicted_class = POTATO_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    casue = POTATO_CAUSE[np.argmax(predictions[0])]
    disc = POTATO_DISC[np.argmax(predictions[0])]
    treat = POTATO_TREAT[np.argmax(predictions[0])]
    prevent = POTATO_PREVENTION[np.argmax(predictions[0])]

    print(predicted_class, confidence)
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': casue,
        'dispcription': disc,
        'treatment': treat,
        'prevention' : prevent
    }
APPLE_CLASS_NAMES = ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"]
MODEL_APPLE = tf.keras.models.load_model("Models/Apple.h5")
APPLE_CAUSE = ["Fungus [Venturia inaequalis]",
               "Fungus [Diplodia seriata]",
               "Pathogen [Gymnosporangium juniperi-virginianae]", 
               "None"]
APPLE_DISC = ["Overwinters on fallen diseased leaves. In spring, these fungi shoot spores into the air. Spores are carried by wind to newly developing leaves, flowers, fruit or green twigs",
              "Overwinters in cankers, mummified fruits, and the bark of dead wood",
              "Reduce yield on apples, blemish the fruit, and lead to weakening and death of redcedar",
               "None"]
APPLE_TREAT = ["Fungicide application must begin in early spring from apple green tip, and continue on a 7- to 10-day schedule (7 days during wet weather, 10 days if dry) until petal fall. If dry weather persists after petal fall, a 10- to 14-day spray schedule is adequate for scab control.",
               "Mancozeb, and Ziram are all highly effective against black rot", 
               "Fungicides with the active ingredient Myclobutanil are most effective in preventing rust, Spray trees and shrubs when flower buds first emerge until spring weather becomes consistently warm and dry, Monitor nearby junipers ",
               "None"]
APPLE_PREVENTION = ["Choose scab-resistant varieties of apple or crabapple trees, Rake up and discard any fallen leaves or fruit on a regular basis, and never leave fallen leaves or fruit on the ground over winter",
               "Prune out dead or diseased branches, Pick all dried and shriveled fruits remaining on the trees, Remove infected plant material from the area, All infected plant parts should be burned, buried or sent to a municipal composting site, Be sure to remove the stumps of any apple trees you cut down", 
               "Control of cedar–apple rust will be best obtained by growing apple varieties that are less susceptible ",
               "None"]

@app.route('/predictApple', methods=['POST'])
def predictApple():
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = image.resize((256,256), Image.LANCZOS)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL_APPLE.predict(img_batch)
    print(predictions)
    predicted_class = APPLE_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    casue = APPLE_CAUSE[np.argmax(predictions[0])]
    disc = APPLE_DISC[np.argmax(predictions[0])]
    treat = APPLE_TREAT[np.argmax(predictions[0])]
    prevent = APPLE_PREVENTION[np.argmax(predictions[0])]
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': casue,
        'dispcription': disc,
        'treatment': treat,
        'prevention' : prevent
    }


MODEL_CHERRY = tf.keras.models.load_model("Models/Cherry.h5")
CHERRY_CLASS_NAMES = ["Powdery Mildew", "Healthy"]
CHERRY_CAUSE = ["[Pathogen] Podosphaera clandestina", "None"]
CHERRY_DISC = ["Mid- and late-season sweet cherry (Prunus avium) cultivars are commonly affected, rendering them unmarketable due to the covering of white fungal growth on the cherry surface", "None"]
CHERRY_TREAT = ["Spray Potassium bicarbonate on plants every one to two weeks", "None"]
CHERRY_PREVENTION = ["Maintain adequate spacing between plants and keep them far enough away from walls and fences to ensure good air circulation and help reduce relative humidity", "None"]

@app.route('/predictCherry', methods=['POST'])
def predictCherry():
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = image.resize((256,256), Image.LANCZOS)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL_APPLE.predict(img_batch)

    predicted_class = CHERRY_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    casue = CHERRY_CAUSE[np.argmax(predictions[0])]
    disc = CHERRY_DISC[np.argmax(predictions[0])]
    treat = CHERRY_TREAT[np.argmax(predictions[0])]
    prevent = CHERRY_PREVENTION[np.argmax(predictions[0])]
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': casue,
        'dispcription': disc,
        'treatment': treat,
        'prevention' : prevent
    }


MODEL_CORN = tf.keras.models.load_model("Models/Corn.h5")
CORN_CLASS_NAMES = ["Gray Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"]
CORN_CAUSE = ["Pathogen [Podosphaera clandestina]",
              "Fungus [Puccinia sorghi]",
              "Fungus [Exserohilum turcicum]", "None",]
CORN_DISC = ["Gray leaf spot requires extended periods of high humidity and warm conditions",
             "Common rust begins with lesions on leaves resembling flecks which develop into small tan spots. These lesions will be found on both the upper and lower surfaces of the leaves or leaf sheaths and are scattered across the leaf surface",
             "Typical symptoms of northern corn leaf blight are canoe-shaped lesions 1 inch to 6 inches long. The lesions are initially bordered by gray-green margins. They eventually turn tan colored and may contain dark areas of fungal sporulation", "None"]
CORN_TREAT = ["During the growing season, foliar fungicides can be used to manage gray leaf spot outbreaks",
                "There is no treatment for rust. Try these tips: Remove all infected parts and destroy them. For bramble fruits, remove and destroy all the infected plants and replant the area with resistant varieties", 
                "By spraying with a mild solution of bicarbonate of soda (baking soda), using ½ teaspoon per gallon (2.5 mL. per 4 L.) of water", "None"]
CORN_PREVENTION = ["Reduce thatch layer. Irrigate deeply, but infrequently. This generally means one time per week with one inch of water. Always irrigate in the morning, which promotes quick drying of the foliage. Avoid using post-emergent weed killers on the lawn while the disease is active.",
                "Dust your plants with sulfur early in the season to prevent infection or to keep mild infections from spreading. Space your plants properly to encourage good air circulation. Avoid wetting the leaves when watering plants.", 
                "Rotating from corn to non-host crops helps reduce favorable environmental conditions for disease pathogens, risk of infection and disease levels..", "None"]

@app.route('/predictCorn', methods=['POST'])
def predictCorn():
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = image.resize((256,256), Image.LANCZOS)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL_APPLE.predict(img_batch)

    predicted_class = CORN_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    casue = CORN_CAUSE[np.argmax(predictions[0])]
    disc = CORN_DISC[np.argmax(predictions[0])]
    treat = CORN_TREAT[np.argmax(predictions[0])]
    prevent = CORN_PREVENTION[np.argmax(predictions[0])]
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': casue,
        'dispcription': disc,
        'treatment': treat,
        'prevention' : prevent
    }



MODEL_GRAPE = tf.keras.models.load_model("Models/Grape.h5")
GRAPE_CLASS_NAMES = ["Black Rot", "Esca", "Leaf Blight", "Healthy", ]
GRAPE_CAUSE = ["Fungus [Guignardia bidwellii]",
               "Fungus [Phaeoacremonium aleophilum]",
               "Bacteria [Xylophilus ampelinus]",
               "None"]
GRAPE_DISC = ["It is one of the most common diseases of grapes in areas where the growing season is warm and humid",
              "Symptoms first become apparent in vineyards 5 to 7 or more years old, but the infections actually occur in younger vines",
              "Bacterial blight of grapevine is a serious, chronic and systemic disease of grapevine that affects commercially important cultivars", "None"]
GRAPE_TREAT = ["Mancozeb, and Ziram are all highly effective against black rot",
               "Captan and Myclobutanil are the fungicides of choice.",
              "Lime sulfur, Sulfur or Copper formulations", "None"]
GRAPE_PREVENTION = ["Before buying crop seeds call your seed company and make sure they can certify that their seed is disease free", "Use alternative pruning methods such as delayed pruning or double pruning. Avoid pruning during periods of heavy rainfall when spores are likely to be dispersed. Monitor the vineyard in Spring, and look for dead spurs or for stunted shoots.",
              "Mulch around the base of the plant with straw, wood chips or other natural mulch to prevent fungal spores in the soil from splashing on the plant.", "None"]

@app.route('/predictGrape', methods=['POST'])
def predictGrape():
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = image.resize((256,256), Image.LANCZOS)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL_APPLE.predict(img_batch)

    predicted_class = GRAPE_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    casue = GRAPE_CAUSE[np.argmax(predictions[0])]
    disc = GRAPE_DISC[np.argmax(predictions[0])]
    treat = GRAPE_TREAT[np.argmax(predictions[0])]
    prevent = GRAPE_PREVENTION[np.argmax(predictions[0])]
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': casue,
        'dispcription': disc,
        'treatment': treat,
        'prevention' : prevent
    }


'''
MODEL_ORANGE = tf.keras.models.load_model("Models/Orange.h5")
ORANGE_CLASS_NAMES = ["Haunglongbing"]
GRAPE_CAUSE = ["fungus Guignardia bidwellii"]
GRAPE_DISC = ["It is one of the most common diseases of grapes in areas where the growing season is warm and humid"]
GRAPE_TREAT = ["The best time to treat black rot of grapes is between bud break until about four weeks after bloom, captan and myclobutanil are the fungicides of choice, Prevention is key when dealing with grape black rot"]
@app.post("/predictOrange")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL_ORANGE.predict(img_batch)

    predicted_class = ORANGE_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    casue = ORANGR_CAUSE[np.argmax(predictions[0])]
    disc = ORANGE_DISC[np.argmax(predictions[0])]
    treat = ORANGE_TREAT[np.argmax(predictions[0])]
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': casue,
        'dispcription': disc,
        'treatment': treat
    }

'''
MODEL_PEACH = tf.keras.models.load_model("Models/Peach.h5")
PEACH_CLASS_NAMES = ["Bacterial Spot", "Healthy"]
PEACH_CAUSE = ["Most Xanthomonas Baterial Family", "None"]
PEACH_DISC = ["Bacterial spot affects peaches, nectarines, apricots, plums, prunes and cherries. The disease is widespread throughout all fruit growing states", "None"]
PEACH_TREAT = ["Copper, Oxytetracycline (Mycoshield and generic equivalents), and Syllit+Captan", "None"]
PEACH_PREVENTION = ["Transplants should be inspected regularly to identify symptomatic seedlings. Transplants with symptoms may be removed and destroyed or treated with streptomycin", "None"]

@app.route('/predictPeach', methods=['POST'])
def predictPeach():
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = image.resize((256,256), Image.LANCZOS)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL_APPLE.predict(img_batch)

    predicted_class = PEACH_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    casue = PEACH_CAUSE[np.argmax(predictions[0])]
    disc = PEACH_DISC[np.argmax(predictions[0])]
    treat = PEACH_TREAT[np.argmax(predictions[0])]
    prevent = PEACH_PREVENTION[np.argmax(predictions[0])]
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': casue,
        'dispcription': disc,
        'treatment': treat,
        'prevention' : prevent
    }


MODEL_PEPPER = tf.keras.models.load_model("Models/Pepper.h5")
PEPPER_CLASS_NAMES = ["Bacterial Spot", "Healthy"]
PEPPER_CAUSE = ["Most Xanthomonas Baterial Family", "None"]
PEPPER_DISC = ["Bacterial spot affects peaches, nectarines, apricots, plums, peppers and cherries. The disease is widespread throughout all fruit growing peppers", "None"]
PEPPER_TREAT = ["Copper, Oxytetracycline (Mycoshield and generic equivalents), and Syllit+Captan;", "None"]
PEPPER_PREVENTION = ["Transplants should be inspected regularly to identify symptomatic seedlings. Transplants with symptoms may be removed and destroyed or treated with streptomycin", "None"]

# @app.post("/predictPepper")
# # async def predict(
# #     file: UploadFile = File(...)
# # ):
@app.route('/predictPepper', methods=['POST'])
def predictPepper():
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = image.resize((256,256), Image.LANCZOS)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL_APPLE.predict(img_batch)

    predicted_class = PEPPER_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    casue = PEPPER_CAUSE[np.argmax(predictions[0])]
    disc = PEPPER_DISC[np.argmax(predictions[0])]
    treat = PEPPER_TREAT[np.argmax(predictions[0])]
    prevent = PEPPER_PREVENTION[np.argmax(predictions[0])]
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': casue,
        'dispcription': disc,
        'treatment': treat,
        'prevention' : prevent
    }




'''MODEL_SQUASH = tf.keras.models.load_model("Models/Squash.h5")
SQUASH_CLASS_NAMES = ["Powdery Mildew"]
@app.post("/predictSquash")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL_SQUASH.predict(img_batch)

    predicted_class = SQUASH_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
casue = SQUASH_CAUSE[np.argmax(predictions[0])]
    disc = SQUASH_DISC[np.argmax(predictions[0])]
    treat = SQUASH_TREAT[np.argmax(predictions[0])]
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': casue,
        'dispcription': disc,
        'treatment': treat
    }
'''

MODEL_STRAWBERRY = tf.keras.models.load_model("Models/Strawberry.h5")
STRAWBERRY_CLASS_NAMES = [ "Leaf Scorch", "Health"]
STRAWBERRY_CAUSE = ["Fungus [Diplocarpon Earlianum]", "None"]
STRAWBERRY_DISC = ["This ascomycete produces disk-shaped, dark brown to black apothecia (0.25-1 mm) on advanced-stage lesions on strawberry leaves and leaf residues (Heidenreich and Turechek).", "None"]
STRAWBERRY_TREAT = ["No Known Cure", "None"]
STRAWBERRY_PREVENTION = ["Prevention of scorch needs to begin with winter watering. A deep soaking once a month, when there is no snow cover, will help prevent root die-back due to dehydration.", "None"]
# @app.post("/predictStrawberry")
# async def predict(
#     file: UploadFile = File(...)
# ):
@app.route('/predictStrawberry', methods=['POST'])
def predictStrawberry():
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = image.resize((256,256), Image.LANCZOS)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL_APPLE.predict(img_batch)

    predicted_class = STRAWBERRY_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    casue = STRAWBERRY_CAUSE[np.argmax(predictions[0])]
    disc = STRAWBERRY_DISC[np.argmax(predictions[0])]
    treat = STRAWBERRY_TREAT[np.argmax(predictions[0])]
    prevent = STRAWBERRY_PREVENTION[np.argmax(predictions[0])]
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': casue,
        'dispcription': disc,
        'treatment': treat,
        'prevention' : prevent
    }

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)




