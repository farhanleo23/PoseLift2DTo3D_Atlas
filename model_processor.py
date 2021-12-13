import os
import numpy as np
import sys

sys.path.append('../')

from acl_model import Model



class ModelProcessor:
    
    def __init__(self, acl_resource, params):
        self._acl_resource = acl_resource
        self.params = params

        assert 'model_dir' in params and params['model_dir'] is not None, 'Review your param: model_dir'
        assert os.path.exists(params['model_dir']), "Model directory doesn't exist {}".format(params['model_dir'])
            
        # load model from path, and get model ready for inference
        self.model = Model(acl_resource, params['model_dir'])

'''
    def predict(self, img_original):
        
        #preprocess image to get 'model_input'
        model_input = self.preprocess(img_original)

        # execute model inference
        infer_output = self.model.execute([model_input]) 

        # postprocessing: 
        category = self.post_process(infer_output)

        return category
'''


