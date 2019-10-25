'''

Ensembling ImageNet models for streaming DL applications.

Author: Prathamesh Mandke

Date created: 10/25/2019

'''

import os
import time
from datetime import timedelta

import torch


class Ensemble:

    def __init__(self, model_path, ensemble_type='avg'):
        ''' Initialize an Ensemble object.

        model_path: path to a directory that contains the models to be ensembled. Ensure that the path contains nothing else other than the models.

        ensemble_type: 'avg' is the only one supported right now. It averages th results ffrom all ensembles.

        '''

        self.ensemble_type = ensemble_type;

        self.model_path = model_path

        self.ensemble_score = None


    def get_prediction(self, input, model, path=False):
        '''path: True if model variable passed above is a path instead the model instance'''
        if path:
            try:
                if self.model_path[-1] == '/':
                    model = torch.load(self.model_path + _)
                else:
                    model = torch.load(self.model_path + '/' + _)
            except:
                print("error reading {}".format(_))

        return model(input)

    def predict_ensemble(self, input):

        self.num_ensembles = 0.0

        for idx, _ in enumerate(os.listdir(self.model_path)):

            torch.cuda.empty_cache()

            print("Forward propagating {}...".format(_))
            t1 = time.monotonic()
            try:
                if self.model_path[-1] == '/':
                    model = torch.load(self.model_path + _)
                else:
                    model = torch.load(self.model_path + '/' + _)
            except:
                print("error reading {}".format(_))
                continue

            if idx:
                self.ensemble_score += self.get_prediction(input, model)
            else:
                self.ensemble_score = self.get_prediction(input, model)
                
            self.num_ensembles += 1.0

            print("Time taken = {}s".format(timedelta(seconds=time.monotonic() - t1)))

        if self.ensemble_type == 'avg':
            return torch.max(self.ensemble_score/self.num_ensembles, 1)[1].item()
