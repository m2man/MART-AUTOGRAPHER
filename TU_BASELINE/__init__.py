import os
import os.path as osp

parent_path = osp.dirname(os.__file__)
dataset_module_path = osp.join(parent_path, 'dataset')
__path__.append(dataset_module_path)
execfile(osp.join(dataset_module_path, '__init__.py'))