import os
import numpy as np
import tensorflow as tf
from parse_data import ParseData
from build_model import BuildModel

BATCH_SIZE = 1024
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_CLASSES = 5

if __name__ == '__main__':
	project_parent_dir = '/home/jyoun/Jason/Classes/ECS289G/RFML'
	data_parent_dir = os.path.join(project_parent_dir, 'Data')
	log_parent_dir = os.path.join(project_parent_dir, 'Log')
	class_list = ['A', 'B', 'C', 'D', 'E']

	# parse
	data_parser = ParseData(data_parent_dir, class_list, NUM_CLASSES)
	train_files_and_labels_np, test_files_and_labels_np = data_parser.split_train_test(train_ratio=0.8)

	# build model
	lstm_model = BuildModel(NUM_CLASSES)
	lstm_model.build()

	loss = lstm_model.loss()
	tf.summary.scalar('loss', loss)

	optimizer = lstm_model.optimizer(loss, LEARNING_RATE)

	accuracy = lstm_model.accuracy()
	tf.summary.scalar('accuracy', accuracy)

	merged = tf.summary.merge_all()

	# init
	iteration = 0
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		# writer
		train_writer = tf.summary.FileWriter(os.path.join(log_parent_dir, 'train'), sess.graph)
		test_writer = tf.summary.FileWriter(os.path.join(log_parent_dir, 'test'), sess.graph)

		# Run the initializer
		sess.run(init)

		for epoch in range(NUM_EPOCHS):
			print('****** Epoch: {}/{} ******'.format(epoch, NUM_EPOCHS))

			total_batch = int(np.ceil(train_files_and_labels_np.shape[0] / BATCH_SIZE))

			# shuffle the training data for each epoch
			np.random.shuffle(train_files_and_labels_np)

			# iteration
			for i in range(total_batch):
				# get corrupted batch using the un-corrupted data_train
				start_idx = i*BATCH_SIZE
				end_idx = (i+1)*BATCH_SIZE
				batch_X, batch_Y = data_parser.get_actual_data_and_labels(train_files_and_labels_np[start_idx:end_idx])

				if iteration % 5 == 0:
					train_summary, current_loss, current_accuracy = sess.run([merged, loss, accuracy], feed_dict={lstm_model.X: batch_X, lstm_model.Y: batch_Y})
					train_writer.add_summary(train_summary, iteration)
					print('({}/{}) loss: {}, accuracy: {}'.format(i, total_batch, current_loss, current_accuracy))

					random_idx = np.random.choice(test_files_and_labels_np.shape[0], BATCH_SIZE)
					test_X, test_Y = data_parser.get_actual_data_and_labels(test_files_and_labels_np[random_idx])
					test_summary = sess.run([merged], feed_dict={lstm_model.X: test_X, lstm_model.Y: test_Y})
					test_writer.add_summary(test_summary, iteration)

				_ = sess.run([optimizer], feed_dict={lstm_model.X: batch_X, lstm_model.Y: batch_Y})

				iteration += 1

		train_writer.close()
		test_writer.close()
