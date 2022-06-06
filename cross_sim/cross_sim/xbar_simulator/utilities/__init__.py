'''
Various utility classes from various external sources (with their own licenses)
'''

try:
    #Original version that uses CC-BY-SA-3.0
    from .viral_license.stackoverflow import DocStringInheritorAbstractBaseClass
except ImportError:
    #An alternative that does not use CC-BY-SA-3.0
    from .alternative import DocStringInheritorAbstractBaseClass

__all__ = ['DocStringInheritorAbstractBaseClass']
