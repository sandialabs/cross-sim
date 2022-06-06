#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# Learning rate schedule
def adjust_learning_rate(lr,epoch):

	# Default behavior: multiply LR by 0.75 ever 5 epochs
	if epoch > 0 and epoch % 5 == 0:
		lr0 = lr
		lr = lr * 0.75
		print('Reduced learning rate from {:.4f}'.format(lr0) +' to {:.4f}'.format(lr))

	return lr