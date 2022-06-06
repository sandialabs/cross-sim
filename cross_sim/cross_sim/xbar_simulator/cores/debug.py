#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

'''
Used to declare :py:data:`.DEBUG_CALLS`, and :py:func:`print_debug_calls`
'''
DEBUG_CALLS = False
'''
Set to :py:data:`True` if you want calls made to various cores to be subject to debug output
'''
# DEBUG_CALLS = True

def print_debug_calls(*args, **kwargs):
    '''
    Prints an output, but only if :py:data:`.DEBUG_CALLS` is enabled
    '''
    if DEBUG_CALLS:
        print(*args, **kwargs)
