'''
Contains an alternative for the code from stackoverflow.py.

This file does not include any non-Sandia code.
'''
from abc import ABCMeta

__all__ = ['DocStringInheritorAbstractBaseClass']

class DocStringInheritorAbstractBaseClass(ABCMeta):
    '''
    This was intended to be a metaclass that allows subclasses to inherit 
    docstrings from superclasses.
    
    Since all of those classes were themselves metaclasses, this class inherits 
    from abc.ABCMeta.
    
    If you want to have proper documentation generated for subclasses, you can 
    copy the __new__ method from http://stackoverflow.com/a/8101118 into this 
    class.
    
    Before distributing this file with the code from StackOverflow, you must 
    ensure that the license of the code from StackOverflow does not conflict 
    with the license of the rest of CrossSim.
    '''
    pass
