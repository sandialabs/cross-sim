#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import copy
from enum import Enum


class ParametersBase(object):
    # set to help IDE, can be commented out
    if False:
        from cross_sim import Parameters
        param_root = Parameters

    #default the readonly override to false (for the parameter class)
    override_readonly=False

    def __init__(self,param_root, **kwargs):

        #save a reference to the object that stores all parameters
        # self.force_set("param_root", weakref.ref(param_root))  # a working weakref would be better
        self.force_set("param_root", param_root)

        # next run the settattr function on all the arguments to run any checks and post processing, i.e. post_set and validate
        for key, value in kwargs.items():
            object.__setattr__(self, key, value) #__set__  in Parameter class will raise an error if the value is readonly


    def __setattr__(self, key, value):


        # check if the attribute already exists to avoid typos
        if hasattr(self,key):
            #check if trying to assign a value to a container object rather than the final parameter
            if isinstance(getattr(self,key),ParametersBase) and not isinstance(value, ParametersBase):
                raise ValueError('Cannot assign a value to '+str(key))
            object.__setattr__(self, key, value)  #will call __set__ in Parameter descriptor
        else:
            raise ValueError("The parameter "+str(key)+" is not defined")

    def force_set(self, key, value):
        # force a value and ignore checks
        object.__setattr__(self, key, value)

    def __deepcopy__(self, memo):
        """
        overwrite the deep copy function to prevent copying the param_root and keeping a reference to it
        MUST call change_param_root if you want to change the param_root to the new object

        :param memo: dictionary of copied values to prevent infinite recursion
        :return:
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for key, value in self.__dict__.items():
            if key == "param_root":
                # result.__dict__[key]=copy.deepcopy(value,memo)
                result.__dict__[key]=self.param_root
            else:
                # can directly store to the dict as Parameter descriptor stores values in the dict
                result.__dict__[key]=copy.deepcopy(value,memo)

        return result



    @staticmethod
    def generate_enum_post_set(name, enum_type):
        """
        Creates a post set function for enum variables.
        Allows text values to be used when assigning a variable and stores an enum value

        :param name:
        :param enum_type:
        :type enum_type: Enum
        :return:
        """
        def post_set(instance):
            obj = getattr(instance,name)
            if isinstance(obj, enum_type ):
                pass
            else:
                if isinstance(obj,str) and hasattr(enum_type, obj):
                    setattr(instance,name,enum_type[obj])
                else:
                    raise ValueError(str(name)+" must be set to one of "+str(list(enum_type.__members__.keys())) )
            pass

        return post_set


class Parameter(object):
    '''
    Holds metadata about the parameter, and implements __get__ and __set__ for the parameter
    Using descriptors adds extra complexity as it defines a property of the class and not of the instance

    '''


    def __init__(self, name, readonly=False, post_set=None):
        '''

        the calling instance must set an instance variable override_readonly=True in order to override readonly

        :param name:  The name of the parameter, this is used so that the value can be stored to the particular instance
        :param readonly: Only allow modifications from within instance methods of the containing class, meant for derived parameters
        :type readonly: bool
        :param post_set: Function called after setting the value (e.g.: to update other read-only parameters)
        :type post_set: callable(instance,name,value)
        '''


        self.name = name
        '''
        This is the name of the parameter.  It is used to store the value in an instance's dict
        '''


        # the following values are shared among all instances of a class
        self.readonly = readonly
        '''
        The value for this parameter can only be modified from its instance.

        (This is checked by making sure that the first parameter of function that requests the modification is named :code:`self`, and that :code:`self` is the name of the instance to which this parameter is attached.
        '''
        # self.validate = validate
        # '''
        # A function to validate new values for the parameter
        # '''
        self.post_set = post_set
        '''
        A function to call after setting a new value (e.g.: to update read-only parameters)
        '''


    def __get__(self, instance, type=None):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)


    def __set__(self, instance, value):

        # check if the readonly has been overridden
        if not getattr(instance, "override_readonly", False):
            if self.readonly:
                raise AttributeError("can't set attribute "+str(self.name))

        instance.__dict__[self.name]=value # store value in instance's dict with same name as descriptor (can only be accessed by directly calling__dict__

        #run post_set
        if self.post_set is not None:
            try:
                self.post_set(instance)
            except AttributeError:
                print('post_set error ',self)
                raise AttributeError