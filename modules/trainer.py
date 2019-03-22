# trainer.py
# Contains functions for training models


import numpy as np
import time
import torch

from visualizer import plotPerformance
from helper import getModelName
from metrics import calculateTotalIoU, getLoss


def trainModel(model, train_loader, valid_loader, batch_size=64, lr=0.001, num_epochs=30, out_suppress=False, checkpoint=True, save_model=True, device=None):
	name = getModelName(lr, batch_size)

	# Arrays to store loss and accuracy for plotting
	train_loss = np.zeros(num_epochs)
	valid_loss = np.zeros(num_epochs)
	train_acc = np.zeros(num_epochs)
	valid_acc = np.zeros(num_epochs)

	# Use BCE loss for pixelwise binary classification problem
	criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	start_time = time.time()

	for epoch in range(num_epochs):
		for i, batch in enumerate(train_loader):
			# Extract data images and target masks
			img, target = batch

			img = img.to(device)
			target = target.to(device)

			optimizer.zero_grad()

			# Forward pass
			pred = model(img)

			loss = criterion(pred, target)

			# Backward pass
			loss.backward()

			# Update weights
			optimizer.step()

		# Keep track of accuracy and loss each epoch
		# Training loss over the entire data set is recalculated at
		# the end of each epoch instead of averaging over the minibatches above
		# so that the same weight set is used for each computation,
		# and so it is more comparable with validation loss
		# (which is also computed using the weights at the end of the epoch)
		train_loss[epoch] = getLoss(model, train_loader, criterion, device)
		train_acc[epoch] = calculateTotalIoU(model, train_loader, device)
		valid_loss[epoch] = getLoss(model, valid_loader, criterion, device)
		valid_acc[epoch] = calculateTotalIoU(model, valid_loader, device)

		# Checkpoint model at current epoch
		if (checkpoint == True):
			torch.save(model.state_dict(), name+"_epoch={}".format(epoch))

		# Print out the training information
		if (out_suppress == False):
			print(("Epoch:{}, Train IoU:{:.4f}, Train Loss:{:.4f}"+"|Valid IoU:{:.4f}, Valid Loss:{:.4f}").format(
				epoch, train_acc[epoch], train_loss[epoch], valid_acc[epoch], valid_loss[epoch]))

	# Training end
	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Training Finished")
	print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
	print("\nFinal Training IoU:", train_acc[-1])
	print("\nBest Validation IoU:", np.amax(valid_acc))
	print("On at epoch:", np.argmax(valid_acc))

	# Plot all curves
	if (out_suppress == False):
		plotPerformance(train_loss, valid_loss, train_acc, valid_acc, num_epochs)

	# Save the loss and accuracies
	# in case needed for future plot/comparision
	if (save_model == True):
		np.savetxt("{}_train_loss.csv".format(name), train_loss)
		np.savetxt("{}_val_loss.csv".format(name), valid_loss)
		np.savetxt("{}_train_acc.csv".format(name), train_acc)
		np.savetxt("{}_val_acc.csv".format(name), valid_acc)

