import os
import numpy as np
import sys
sys.path.append('..')
from model_processor import ModelProcessor
import acl
from acl_resource import AclResource

model_path = 'videopose3d2.om'
acl_resource = AclResource()
acl_resource.init()
    
model_parameters = {'model_dir': model_path}
model_processor = ModelProcessor(acl_resource, model_parameters)