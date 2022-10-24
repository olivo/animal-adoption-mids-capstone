import requests
from requests.structures import CaseInsensitiveDict
import json

class petfinder_client:
    def __init__(self, client_id, client_secret):
        self.token_url = "https://api.petfinder.com/v2/oauth2/token"
        self.get_animal_url = "https://api.petfinder.com/v2/animals"
        self.client_id = client_id
        self.client_secret = client_secret
        self.bearer_token = self.get_bearer_token()

    def get_bearer_token(self):
        data = {'grant_type' : 'client_credentials', 'client_id' : self.client_id , 'client_secret' : self.client_secret}
        response = requests.post(self.petfinder_token_url, data = data)
        json_response = json.loads(response.content)

        return json_response['access_token']

    def get_animal(self, id):
        headers = CaseInsensitiveDict()
        headers["Accept"] = "application/json"
        headers["Authorization"] = "Bearer " + self.bearer_token

        params = {'id' : id}

        response = requests.get(self.petfinder_get_animal_url, params = params, headers = headers)

        json_response = json.loads(response.content)

        return json_response

