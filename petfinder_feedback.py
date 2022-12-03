import cv2
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import petfinder_animal
import requests
import shutil

class petfinder_feedback:
    def __init__(self, animal):
        self.sentence_sentiments = self.get_sentence_sentiments(animal.Description)
        # TODO: Need to add Photos field to petfinder_animal. 
        self.photo_blurriness = self.get_photo_blurriness(animal.Photos)

    def get_sentence_sentiments(self, description):
        sentence_to_negative_prob = dict()

        sia = SentimentIntensityAnalyzer()
        sentences = nltk.sent_tokenize(description)
        for sentence in sentences:
            polarity_scores = sia.polarity_scores(sentence)
            sentence_to_negative_prob[sentence] = polarity_scores['neg']

        return sentence_to_negative_prob

    def get_photo_blurriness(self, photos):
        photo_blurriness = dict()

        # Getting all full pictures.
        for photo in photos:
            full_photo_url = photo['full']

            # Download the photo locally.
            res = requests.get(full_photo_url, stream = True)
            with open("temp.jpg",'wb') as f:
                shutil.copyfileobj(res.raw, f)

            # Compute the LaPlacian for the photo.
            imagePath = "temp.jpg"
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Store the result.
            photo_blurriness[full_photo_url] = fm
        
        return photo_blurriness