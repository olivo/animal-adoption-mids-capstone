import cv2
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np
import requests
import shutil

class petfinder_animal:
    def __init__(self, petfinder_attribute_dictionary):
        self.Type = petfinder_attribute_dictionary['type']
        self.Breed1 = petfinder_attribute_dictionary['breeds']['primary']
        self.Breed2 = petfinder_attribute_dictionary['breeds']['secondary']
        self.Gender = petfinder_attribute_dictionary['gender']
        self.Color1 = petfinder_attribute_dictionary['colors']['primary']
        self.Color2 = petfinder_attribute_dictionary['colors']['secondary']
        self.Color3 = petfinder_attribute_dictionary['colors']['tertiary']
        self.MaturitySize = petfinder_attribute_dictionary['size']
        #self.FurLength = 0 # Not specified.
        self.FurLength = None
        self.Vaccinated = petfinder_attribute_dictionary['attributes']['shots_current']
        #self.Dewormed = 3 # Not sure.
        self.Dewormed = None
        self.Sterilized = petfinder_attribute_dictionary['attributes']['spayed_neutered']
        #self.Health = 0 # Not specified
        self.Health = None
        #self.State = 0 # Unknown
        self.State = petfinder_attribute_dictionary['contact']['address']['state']
        self.Description = petfinder_attribute_dictionary['description']
        self.HasDescription = self.Description != None and self.Description != ""
        self.Age = petfinder_attribute_dictionary['age']
        self.Quantity = None
        self.Fee = None
        self.VideoAmt = len(petfinder_attribute_dictionary['videos'])
        self.PhotoAmt = len(petfinder_attribute_dictionary['photos'])
        self.DescriptionLength = len(self.Description) if self.HasDescription else 0
        self.Photos = petfinder_attribute_dictionary['photos']
        self.Url = petfinder_attribute_dictionary['url']

    def create_harmonized_petfinder_animal(petfinder_attribute_dictionary):
        animal = petfinder_animal(petfinder_attribute_dictionary)
        animal.harmonize_fields()
        animal.add_sentiment_fields()
        animal.add_image_fields()

        return animal

    def harmonize_fields(self):
        self.Type = self.harmonized_Type(self.Type)
        self.Breed1 = self.harmonized_Breed(self.Breed1)
        self.Breed2 = self.harmonized_Breed(self.Breed2)
        self.Gender = self.harmonized_Gender(self.Gender)
        self.Color1 = self.harmonized_Color(self.Color1)
        self.Color2 = self.harmonized_Color(self.Color2)
        self.Color3 = self.harmonized_Color(self.Color3)
        self.Description = self.harmonized_Description(self.Description)
        self.MaturitySize = self.harmonized_MaturitySize(self.MaturitySize)
        self.FurLength = self.harmonized_FurLength(self.FurLength)
        self.Vaccinated = self.harmonized_Vaccinated(self.Vaccinated)
        self.Dewormed = self.harmonized_Dewormed(self.Dewormed)
        self.Sterilized = self.harmonized_Sterilized(self.Sterilized)
        self.Health = self.harmonized_Health(self.Health)
        self.State = self.harmonized_State(self.State)
        self.Age = self.harmonized_Age(self.Age)
        self.Quantity = self.harmonized_Quantity(self.Quantity)
        self.Fee = self.harmonized_Fee(self.Fee)

    # Age in the live Petfinder dataset is a category (such as 'Puppy', 'Young', 'Adult'), rather
    # than age in months as in the Petfinder competition.
    def harmonized_Age(self, age):
        return 0

    # Note that this harmonization is incomplete, as there are 300 breeds in the petfinder dataset.
    # We should create a dictionary from the breeds label.
    def harmonized_Breed(self, breed):
        breed_dictionary = {
            'Affenpinscher' : 1,
            'Afghan Hound' : 2,
            'Airedale Terrier' : 3,
            'Akbash' : 4,
            'Akita' : 5,
            'Alaskan Malamute' : 6,
            'American Bulldog' : 7,
            'American Eskimo Dog' : 8,
            'American Hairless Terrier' : 9,
            'American Staffordshire Terrier' : 10,
            'Australian Shepherd' : 16
        }
        
        return breed_dictionary.get(breed, 0)

    def harmonized_Color(self, color):
        color_dictionary = {
            'Black' : 1,
            'Brown' : 2,
            'Golden' : 3,
            'Yellow' : 4,
            'Cream' : 5,
            'Gray' : 6,
            'White' : 7
        }

        return color_dictionary.get(color, 0)

    def harmonized_Description(self, description):
        if description is None:
            return ""
        else:
            return description

    def harmonized_Dewormed(self, Dewormed):
        return 3 # 3: Not sure

    # Currently Petfinder is not showing pet fees in the profile pages.
    def harmonized_Fee(self, Fee):
        return 0

    def harmonized_FurLength(self, furLength):
        return 0

    def harmonized_Gender(self, gender):
        if gender == 'Male':
            return 1
        elif gender == 'Female':
            return 2
        else:
            return None

    def harmonized_Health(self, health):
        return 0 # 0: Not specified

    def harmonized_MaturitySize(self, maturitySize):
        if maturitySize == 'Small':
            return 1
        elif maturitySize == 'Medium':
            return 2
        elif maturitySize == 'Large':
            return 3
        elif maturitySize == 'Extra Large':
            return 4
        else:
            return 0

    # All live profiles in Petfinder seem to be for one pet.
    def harmonized_Quantity(self, quantity):
        return 1

    # Note: In the Kaggle dataset states are from Malaysia, so we need to train/test on
    # the same state space to have a meaningful value.
    def harmonized_State(self, state):
        return 0

    def harmonized_Sterilized(self, sterilized):
        if sterilized:
            return 1
        else:
            return 0

    def harmonized_Type(self, type):
        if type == 'Dog':
            return 1
        elif type == 'Cat':
            return 2
        else:
            return None

    def harmonized_Vaccinated(self, vaccinated):
        if vaccinated:
            return 1
        else:
            return 2

    def add_sentiment_fields(self):
        if self.HasDescription:
            sia = SentimentIntensityAnalyzer()
            polarity_scores = sia.polarity_scores(self.Description)

            self.nltk_negative_prob = polarity_scores['neg']
            self.nltk_neutral_prob = polarity_scores['neu']
            self.nltk_positive_prob = polarity_scores['pos']
            self.nltk_compound_score = polarity_scores['compound']
        else:
            self.nltk_negative_prob = 0
            self.nltk_neutral_prob = 0
            self.nltk_positive_prob = 0
            self.nltk_compound_score = 0

    def add_image_fields(self):
        if self.Photos == []:
            self.AvgLaPlacianVariance = 0
            return 0

        photo_blurriness = []
        for photo in self.Photos:
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
            photo_blurriness.append(fm)

        self.AvgLaPlacianVariance = np.mean(photo_blurriness)      

    def as_dictionary(self):
        property_dictionary = {
            'Type' : self.Type,
            'Breed1' : self.Breed1,
            'Breed2' : self.Breed2,
            'Gender' : self.Gender,
            'Color1' : self.Color1,
            'Color2' : self.Color2,
            'Color3' : self.Color3,
            'MaturitySize' : self.MaturitySize,
            'FurLength' : self.FurLength,
            'Vaccinated' : self.Vaccinated,
            'Dewormed' : self.Dewormed,
            'Sterilized' : self.Sterilized,
            'Health' : self.Health,
            'State' : self.State,
            'Description' : self.Description,
            'HasDescription' : self.HasDescription,
            'Age' : self.Age,
            'Quantity' : self.Quantity,
            'Fee' : self.Fee,
            'VideoAmt' : self.VideoAmt,
            'PhotoAmt' : self.PhotoAmt,
            'DescriptionLength' : self.DescriptionLength,
            'nltk_negative_prob' : self.nltk_negative_prob,
            'nltk_neutral_prob' : self.nltk_neutral_prob,
            'nltk_positive_prob' : self.nltk_positive_prob,
            'nltk_compound_score' : self.nltk_compound_score,
            'AvgLaPlacianVariance' : self.AvgLaPlacianVariance
        }

        return property_dictionary

    def as_kaggle_dictionary(self):
        property_dictionary = {
            'Type' : self.Type,
            'Breed1' : self.Breed1,
            'Breed2' : self.Breed2,
            'Gender' : self.Gender,
            'Color1' : self.Color1,
            'Color2' : self.Color2,
            'Color3' : self.Color3,
            'MaturitySize' : self.MaturitySize,
            'FurLength' : self.FurLength,
            'Vaccinated' : self.Vaccinated,
            'Dewormed' : self.Dewormed,
            'Sterilized' : self.Sterilized,
            'Health' : self.Health,
            'State' : self.State,
            'HasDescription' : self.HasDescription,
            'Age' : self.Age,
            'Quantity' : self.Quantity,
            'Fee' : self.Fee,
            'VideoAmt' : self.VideoAmt,
            'PhotoAmt' : self.PhotoAmt,
            'DescriptionLength' : self.DescriptionLength,
            'nltk_negative_prob' : self.nltk_negative_prob,
            'nltk_neutral_prob' : self.nltk_neutral_prob,
            'nltk_positive_prob' : self.nltk_positive_prob,
            'nltk_compound_score' : self.nltk_compound_score,
            'AvgLaPlacianVariance' : self.AvgLaPlacianVariance
        }

        return property_dictionary