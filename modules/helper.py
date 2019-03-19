# helper.py
# Contains miscellaneous helper functions


# Generates name for saving from hyperparameters
def getModelName(lr, batch_size):
	return "model_lr={}_bs={}".format(lr, batch_size)

