from keras.models import Sequential
from keras.layers import Conv2D
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy
import cv2
from text_OCR_detection_helper import predict_model
import argparse

def improve_img_quality(image):
	srcnn_model = predict_model()
	#print("[INFO] Carregando pesos para modelo de aprimoramento da qualidade da imagem ...\n")
	srcnn_model.load_weights("assets/improve_quality_model.h5")
	img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
	print("[INFO] Realizando processo de aprimoramento da qualidade da imagem ...\n")
	Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
	Y[0, :, :, 0] = img[:, :, 0].astype(float) / 255.
	pre = srcnn_model.predict(Y, batch_size=1) * 255.
	pre[pre[:] > 255] = 255
	pre[pre[:] < 0] = 0
	pre = pre.astype(numpy.uint8)
	img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
	img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
	#print("[INFO] Retornando imagem aprimorada ...\n")
	return img
	
if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_image", required=True,
                    help="input pdf name")
    ap.add_argument("-o", "--output_image", type=str,
                    default="improved_image.png",
                    help="path to output image")
    args = vars(ap.parse_args())	
    image = cv2.imread(args["input_image"], cv2.IMREAD_COLOR)
    img = improve_img_quality(image)
    cv2.imwrite(args["output_image"], img)
