import lstm
import time
import matplotlib.pyplot as plt
from keras.models import load_model

def plot_results(predicted_data, true_data, fileName):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig(fileName)

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
        plt.savefig('multipleResults.jpg')


def trainModel(newModel, epochs=1, seq_len=50):
	'''
	Trains and saves model
	'''
	if newModel:
		print('> Data Loaded. Compiling LSTM model...')

		model = lstm.build_model([1, 50, 100, 1])

		model.save('./model/lstm.h5')
	else:
		print('> Data Loaded. Loading LSTM model...')

		model = load_model('./model/lstm.h5')

	return model



#Main Run Thread
if __name__=='__main__':
	global_start_time = time.time()

	# Parameters
	stockFile = './data/lstm/Google.csv'
	epochs = 1
	seq_len = 50
	batch_size=512

	print('> Loading data... ')

	X_train, y_train, X_test, y_test = lstm.load_data(stockFile, seq_len, True)

	# Train and return the model
	model = trainModel(False)

	print('> LSTM trained, Testing model on validation set... ')

	hist = model.fit(
	    X_train,
	    y_train,
	    batch_size=batch_size,
	    nb_epoch=epochs,
	    validation_split=0.05)

	print('> Training duration (s) : ', time.time() - global_start_time)

	print('> Plotting point by point prediction....')
	predicted = lstm.predict_point_by_point(model, X_test)
	plot_results(predicted, y_test, 'ppResults.jpg')

	print('> Plotting full sequence prediction....')
	predicted = lstm.predict_sequence_full(model, X_test, seq_len)
	plot_results(predicted, y_test, 'sResults.jpg')

	print('> Plotting multiple sequence prediction....')
	predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
	plot_results_multiple(predictions, y_test, 50)
