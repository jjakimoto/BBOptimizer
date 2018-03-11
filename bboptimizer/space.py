# -*- coding: utf-8 -*-
# @Author: tom-hydrogen
# @Date:   2018-03-02 14:59:58
# @Last Modified by:   tom-hydrogen
# @Last Modified time: 2018-03-07 14:53:40
# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)\
import numpy as np
from copy import deepcopy

from .exceptions import InvalidConfigError
from .variables import create_variable


class DesignSpace(object):
    """
    Class to handle the input domain of the function.
    The format of a input domain, possibly with restrictions:
    The domain is defined as a list of dictionaries contains a list of attributes, e.g.:
    - Continuous domain
    space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
             {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':2},
             {'name': 'var_3', 'type': 'bandit', 'domain': [(-1,1),(1,0),(0,1)], 'dimensionality':2},
             {'name': 'var_4', 'type': 'bandit', 'domain': [(-1,4),(0,0),(1,2)]},
             {'name': 'var_5', 'type': 'discrete', 'domain': (0,1,2,3)}]
    - Discrete domain
    space =[ {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]
             {'name': 'var_3', 'type': 'discrete', 'domain': (-10,10)}]
    - Mixed domain
    space =[{'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality' :1},
            {'name': 'var_4', 'type': 'continuous', 'domain':(-3,1), 'dimensionality' :2},
            {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]
    Restrictions can be added to the problem. Each restriction is of the form c(x) <= 0 where c(x) is a function of
    the input variables previously defined in the space. Restrictions should be written as a list
    of dictionaries. For instance, this is an example of an space coupled with a constraint
    space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality' :2}]
    constraints = [ {'name': 'const_1', 'constraint': 'x[:,0]**2 + x[:,1]**2 - 1'}]
    If no constraints are provided the hypercube determined by the bounds constraints are used.
    Note about the internal representation of the vatiables: for variables in which the dimaensionality
    has been specified in the domain, a subindex is internally asigned. For instance if the variables
    is called 'var1' and has dimensionality 3, the first three positions in the internal representation
    of the domain will be occupied by variables 'var1_1', 'var1_2' and 'var1_3'. If no dimensionality
    is added, the internal naming remains the same. For instance, in the example above 'var3'
    should be fixed its original name.
    param space: list of dictionaries as indicated above.
    param constraints: list of dictionaries as indicated above (default, none)
    """
    supported_types = ['continuous', 'integer', 'discrete', 'categorical']

    def __init__(self, space, store_noncontinuous=False):

        ## --- Complete and expand attributes
        self.store_noncontinuous = store_noncontinuous
        self.config_space = space

        ## --- Transform input config space into the objects used to run the optimization
        self._translate_space(self.config_space) # Build expanded configuration
        self._expand_space() # Build self.config_space_expanded and self.space_expanded
        self._compute_variables_indices() # Set index

        ## -- Compute raw and model dimensionalities
        self.objective_dimensionality = len(self.space_expanded)
        self.model_input_dims = [v.dimensionality_in_model for v in self.space_expanded]
        self.model_dimensionality = sum(self.model_input_dims)

    def _compute_variables_indices(self):
        """
        Computes and saves the index location of each variable (as a list) in the objectives
        space and in the model space. If no categorical variables are available, these two are
        equivalent.
        """

        counter_objective = 0
        counter_model = 0

        for variable in self.space_expanded:
            variable.set_index_in_objective([counter_objective])
            counter_objective += 1

            if variable.type is not 'categorical':
                variable.set_index_in_model([counter_model])
                counter_model += 1
            else:
                num_categories = len(variable.domain)
                variable.set_index_in_model(list(range(counter_model,counter_model + num_categories)))
                counter_model += num_categories

    def _expand_config_space(self):
        """
        Expands the config input space into a list of dictionaries, one for each variable_dic
        in which the dimensionality is always one.
        Example: It would transform
        config_space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
                        {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':2},
        into
        config_expande_space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
                      {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':1},
                      {'name': 'var_2_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':1}]
        """
        self.config_space_expanded = []

        for variable in self.config_space:
            variable_dic = variable.copy()
            if 'dimensionality' in variable_dic.keys():
                dimensionality = variable_dic['dimensionality']
                variable_dic['dimensionality'] = 1
                variables_set = [variable_dic.copy() for d in range(dimensionality)]
                k = 1
                for variable in variables_set:
                    variable['name'] = variable['name'] + '_' + str(k)
                    k += 1
                self.config_space_expanded += variables_set
            else:
                self.config_space_expanded += [variable_dic]

    def _translate_space(self, space):
        """
        Translates a list of dictionaries into internal list of variables
        """
        self.space = []
        self.dimensionality = 0
        self.has_types = d = {t: False for t in self.supported_types}

        for i, d in enumerate(space):
            descriptor = deepcopy(d)
            descriptor['name'] = descriptor.get('name', 'var_' + str(i))
            descriptor['type'] = descriptor.get('type', 'continuous')
            if 'domain' not in descriptor:
                raise InvalidConfigError('Domain attribute is missing for variable ' + descriptor['name'])
            variable = create_variable(descriptor)
            self.space.append(variable)
            self.dimensionality += variable.dimensionality
            self.has_types[variable.type] = True

    def _expand_space(self):
        """
        Creates an internal list where the variables with dimensionality larger than one are expanded.
        This list is the one that is used internally to do the optimization.
        """

        ## --- Expand the config space
        self._expand_config_space()

        ## --- Expand the space
        self.space_expanded = []
        for variable in self.space:
            self.space_expanded += variable.expand()

    def objective_to_model(self, x_objective):
        ''' This function serves as interface between objective input vectors and
        model input vectors'''

        x_model = []
        for k in range(self.objective_dimensionality):
            variable = self.space_expanded[k]
            new_entry = variable.objective_to_model(x_objective[0][k])
            x_model += new_entry

        return np.array(x_model)

    def model_to_objective(self, x_model):
        ''' This function serves as interface between model input vectors and
            objective input vectors
        '''
        idx_model = 0
        x_objective = []

        for idx_obj in range(self.objective_dimensionality):
            variable = self.space_expanded[idx_obj]
            new_entry = variable.model_to_objective(x_model, idx_model)
            x_objective += new_entry
            idx_model += variable.dimensionality_in_model

        return x_objective

    def get_bounds(self):
        """
        Extracts the bounds of all the inputs of the domain of the *model*
        """
        bounds = []

        for variable in self.space_expanded:
            bounds += variable.get_bounds()

        return bounds
