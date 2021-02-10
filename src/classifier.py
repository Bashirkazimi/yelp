"""classifier model for yelp photos dataset"""


# import tensorflow as tf 
import keras


def create_model(input_shape=(None, None, 3), num_classes=5):
	"""creates a classifier model taking input shape and number of categories as arguments."""
	
	# input layer
	inputs = keras.layers.Input(shape=input_shape)

	# MobileNetV2 is the base_model used as feature extractor 
	base_model = keras.applications.MobileNetV2(input_shape=input_shape,include_top=False,weights='imagenet')

	# we only finetune
	base_model.trainable=False

	# creating the whole pipeline
	features = base_model(inputs)
	x = keras.layers.GlobalAveragePooling2D()(features)
	x = keras.layers.Dense(256, activation='relu')(x)
	x = keras.layers.Dropout(rate=0.2)(x)
	x = keras.layers.Dense(num_classes, activation='softmax')(x)

	# model
	model = keras.Model(inputs=inputs, outputs=x)
	return model








