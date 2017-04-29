import lstm
import time
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import sequence

def plot_results(predicted_data, true_data, fileName):
	'''
	Plots prediction dots and true data
	'''
	fig = plt.figure(facecolor='white', figsize=(20,10))
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.savefig(fileName)

def plot_results_multiple(predicted_data, true_data, prediction_len):
	'''
	Plots multiple sequence predictions and true data
	'''
	fig = plt.figure(facecolor='white', figsize=(30,10))
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	#Pad the list of predictions to shift it in the graph to it's correct start
	for i, data in enumerate(predicted_data):
		padding = [None for p in range(i * prediction_len)]
		plt.plot(padding + data, label='Prediction')
		plt.legend()
		plt.savefig('./out/multipleResults.jpg')

def plotMetrics(history):
	'''
	Plots loss and MSE during epochs
	'''
	losses = []
	mses = []
	for key, value in history.items():
		if(key == 'loss'):
			losses = value
	plt.figure(figsize=(6, 3))
	plt.plot(losses)
	plt.ylabel('error')
	plt.xlabel('iteration')
	plt.title('testing error over time')
	plt.savefig('losses.png')

def trainModel(newModel, epochs=1, seq_len=50):
	'''
	Trains and saves model
	'''
	if newModel:
		global_start_time = time.time()

		print('> Data Loaded. Compiling LSTM model...')

		model = lstm.build_model([1, 50, 100, 1])

		model.save('./../model/lstm.h5')

		print('> Training duration (s) : ', time.time() - global_start_time)
	else:
		print('> Data Loaded. Loading LSTM model...')

		model = load_model('./../model/lstm.h5')

	return model

def run():
	'''
	Main method for manual testing
	'''
	# Parameters
	stockFile = './../data/lstm/GOOG.csv'
	epochs = 10
	seq_len = 100
	batch_size=512

	print('> Loading data... ')

	X_train, y_train, X_test, y_test = lstm.load_data(stockFile, seq_len, True)

	X_train = sequence.pad_sequences(X_train, maxlen=seq_len)
	X_test = sequence.pad_sequences(X_test, maxlen=seq_len)

	print('> X_train seq shape: ', X_train.shape)
	print('> X_test seq shape: ', X_test.shape)

	# Train and return the model
	model = trainModel(True)

	#plot_model(model, to_file='model.png')

	print('> LSTM trained, Testing model on validation set... ')

	training_start_time = time.time()

	hist = model.fit(
	    X_train,
	    y_train,
	    batch_size=batch_size,
	    nb_epoch=epochs,
	    validation_split=0.05,
		validation_data=(X_test, y_test))

	print('> Testing duration (s) : ', time.time() - training_start_time)

	score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
	print('Test score:', score)
	print('Test accuracy:', acc)

	print('> Plotting Losses....')
	plotMetrics(hist.history)

	print('> Plotting point by point prediction....')
	predicted = lstm.predict_point_by_point(model, X_test)

	print(predicted)
	#plot_results(predicted, y_test, './out/ppResults.jpg')

	print('> Plotting full sequence prediction....')
	#predicted = lstm.predict_sequence_full(model, X_test, seq_len)
	#plot_results(predicted, y_test, './out/sResults.jpg')

	print('> Plotting multiple sequence prediction....')
	#predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
	#plot_results_multiple(predictions, y_test, 50)

#Main Run Thread
if __name__=='__main__':
	run()


