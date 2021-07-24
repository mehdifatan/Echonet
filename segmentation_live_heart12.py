# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:16:44 2020

@author: AI-Bridge
"""


import SimpleITK as sitk
import numpy as np
from skimage.transform import rescale, resize
import torch
import torchvision
from torch import nn
import os
import matplotlib.pylab as plt
from PIL import ImageFont, ImageDraw, Image
import segmentation_models_pytorch as smp
import time
import cv2
from PIL import Image
import imageio
from skimage import img_as_ubyte

pretrained = False


# loc = 'testing\patient0001\patient0001_2CH_ED.raw'
# loc=r'D:\Projects\Heart\camus\testing\patient0001\patient0001_2CH_ED.mhd'
# loc=r'D:\Projects\Heart\camus\testing\patient0001\patient0001_2CH_sequence.mhd'
loc='/home/ali/bridge/bridge/echonet/testing/patient0001/patient0001_4CH_sequence.mhd'


#------------------------------model 1
modelname="deeplabv3_resnet50"
output = os.path.join("output", "segmentation", "{}".format(modelname))

#-----model definition and loading
model1 = torchvision.models.segmentation.__dict__[modelname](pretrained=pretrained, aux_loss=False)
model1.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # change channel from 3 to 1
n_class = 4
model1.classifier[-1] = torch.nn.Conv2d(model1.classifier[-1].in_channels, n_class, kernel_size=model1.classifier[-1].kernel_size)  # change number of outputs
# model.to(device)
checkpoint = torch.load(os.path.join(output, "best.pt"), map_location=torch.device('cpu'))
model1.load_state_dict(checkpoint['state_dict'])



#------------------------------model 2
modelname="FPN"
output = os.path.join("output", "segmentation", "{}".format(modelname))

#-----model definition and loading
model2=smp.FPN('resnet18', classes=4, encoder_weights='imagenet',activation='softmax')
model2.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # change channel from 3 to 1

# model.to(device)
checkpoint = torch.load(os.path.join(output, "best.pt"), map_location=torch.device('cpu'))
model2.load_state_dict(checkpoint['state_dict'])



#------------------------------model 3
modelname="linknet"
output = os.path.join("output", "segmentation", "{}".format(modelname))

#-----model definition and loading
model3=smp.Linknet('resnet18', classes=4, encoder_weights='imagenet',activation='sigmoid') 
model3.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # change channel from 3 to 1n_class = 4

# model.to(device)
checkpoint = torch.load(os.path.join(output, "best.pt"), map_location=torch.device('cpu'))
model3.load_state_dict(checkpoint['state_dict'])



#------------------------------model 4
modelname="PSPNet"
output = os.path.join("output", "segmentation", "{}".format(modelname))

#-----model definition and loading
model4=smp.PSPNet('resnet18', classes=4, encoder_weights='imagenet',activation='softmax') 
model4.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # change channel from 3 to 1

# model.to(device)
checkpoint = torch.load(os.path.join(output, "best.pt"), map_location=torch.device('cpu'))
model4.load_state_dict(checkpoint['state_dict'])



#------------------------------model 5
modelname="Unet"
output = os.path.join("output", "segmentation", "{}".format(modelname))

#-----model definition and loading
model5=smp.Unet('resnet18', classes=4, encoder_weights='imagenet',activation='softmax')
model5.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # change channel from 3 to 1

# model.to(device)
checkpoint = torch.load(os.path.join(output, "best.pt"), map_location=torch.device('cpu'))
model5.load_state_dict(checkpoint['state_dict'])


#---------------------------------reading sequence files from folders and doing segmentation


loc='/home/ali/bridge/bridge/echonet/testing'


seq_file='patient0001_4CH_sequence.mhd'

folders = os.listdir(loc) 

#print(folders)

for folder in folders:
	
	
	path = os.path.join(loc,  folder, folder+'_4CH_sequence.mhd')
	 
	#------reading movie file
	image = sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))

	print(np.shape(image))



	z,x,y=np.shape(image)
	orig_im1=np.zeros((20,256,256))
	orig_im2=np.zeros((20,256,256))
	orig_im3=np.zeros((20,256,256))
	orig_im4=np.zeros((20,256,256))
	orig_im5=np.zeros((20,256,256))

	seg_im1=np.zeros((20,256,256))
	seg_im2=np.zeros((20,256,256))
	seg_im3=np.zeros((20,256,256))
	seg_im4=np.zeros((20,256,256))
	seg_im5=np.zeros((20,256,256))

	im_all=np.zeros((20,256,1380,3))
	gap=np.zeros((256,25,3))+255
	
	if z>=20:

		k=0
		for i in range(0,20,4):
			im=[]
			img=image[i:i+4,:,:]
			#print(np.shape(img))
			for j in range(4):
				im.append(resize(img[j,:,:],[256,256],mode='constant'))
			im=np.array(im)
			#print(np.shape(im))
			im=np.reshape(im,(4,1,256,256))/255.0
			#print(np.shape(im))
			#im = Image.fromarray(im)
			im=torch.from_numpy(im)
			#im = im.to(device)

			y_pred1 = model1(im)["out"]
			softmax1 = torch.exp(y_pred1).cpu()
			prob1 = list(softmax1.detach().numpy())
			predictions1 = np.argmax(prob1, axis=1)
			seg_im1[k:k+4,:,:]=predictions1[0:4,:,:]

			y_pred2 = model2(im)
			softmax2 = torch.exp(y_pred2).cpu()
			prob2 = list(softmax2.detach().numpy())
			predictions2 = np.argmax(prob2, axis=1)
			seg_im2[k:k+4,:,:]=predictions2[0:4,:,:]

			y_pred3 = model3(im)
			softmax3 = torch.exp(y_pred3).cpu()
			prob3 = list(softmax3.detach().numpy())
			predictions3 = np.argmax(prob3, axis=1)
			seg_im3[k:k+4,:,:]=predictions3[0:4,:,:]

			y_pred4 = model4(im)
			softmax4 = torch.exp(y_pred4).cpu()
			prob4 = list(softmax4.detach().numpy())
			predictions4 = np.argmax(prob4, axis=1)
			seg_im4[k:k+4,:,:]=predictions4[0:4,:,:]

			y_pred5= model5(im)
			softmax5 = torch.exp(y_pred5).cpu()
			prob5 = list(softmax5.detach().numpy())
			predictions5 = np.argmax(prob5, axis=1)
			seg_im5[k:k+4,:,:]=predictions5[0:4,:,:]

			#print(np.shape(predictions))
			orig_im1[k:k+4,:,:]=im[0:4,0,:,:]
			orig_im2[k:k+4,:,:]=im[0:4,0,:,:]
			orig_im3[k:k+4,:,:]=im[0:4,0,:,:]
			orig_im4[k:k+4,:,:]=im[0:4,0,:,:]
			orig_im5[k:k+4,:,:]=im[0:4,0,:,:]

			k=k+4
		#--------------------------------------------------------    
		

		#Define the codec
		#today = time.strftime("%Y%m%d-%H%M%S")
		#fps_out = 32
		#fourcc = cv2.VideoWriter_fourcc(*'XVID')
		#out = cv2.VideoWriter(today + ".avi", fourcc, fps_out, (256, 1380))


		#video = cv2.VideoWriter('segmentation_movie.avi', 0, 1, (256,1380))    
		#import moviepy.video.io.ImageSequenceClip


		font = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (10,30)
		fontScale = 0.5
		fontColor = (255,255,255)
		lineType = 2


		for i in range(20):

			ori1=orig_im1[i,:,:]*255
			seg1=seg_im1[i,:,:]*255

			ori2=orig_im2[i,:,:]*255
			seg2=seg_im2[i,:,:]*255

			ori3=orig_im3[i,:,:]*255
			seg3=seg_im3[i,:,:]*255

			ori4=orig_im4[i,:,:]*255
			seg4=seg_im4[i,:,:]*255

			ori5=orig_im5[i,:,:]*255
			seg5=seg_im5[i,:,:]*255




			cv2.putText(ori1,'DeepLabV3',
			bottomLeftCornerOfText,
			font,
			fontScale,
			fontColor,
			lineType)


			cv2.putText(ori2,'FPN',
			bottomLeftCornerOfText,
			font,
			fontScale,
			fontColor,
			lineType)


			cv2.putText(ori3,'Linknet',
			bottomLeftCornerOfText,
			font,
			fontScale,
			fontColor,
			lineType)


			cv2.putText(ori4,'PSPNet',
			bottomLeftCornerOfText,
			font,
			fontScale,
			fontColor,
			lineType)


			cv2.putText(ori5,'Unet',
			bottomLeftCornerOfText,
			font,
			fontScale,
			fontColor,
			lineType)



			im1 = np.stack((seg1,ori1, ori1), axis=2)
			im2 = np.stack((seg2,ori2, ori2), axis=2)
			im3 = np.stack((seg3,ori3, ori3), axis=2)
			im4 = np.stack((seg4,ori4, ori4), axis=2)
			im5 = np.stack((seg5,ori5, ori5), axis=2)


			im=np.concatenate((im1,gap,im2,gap,im3,gap,im4,gap,im5),axis=1)


			im2=im.astype(np.uint8)


			cv2.imshow('Segmentation of Heart Cardiac Images', im2)


			#print(im[100,120])

			#im=im*255

			#im2=im.astype(np.uint8)

			#print(im2[100,120])
			#im2 = Image.fromarray(im)

			im_all[i,:,:,:]=im2


			#im_all = img_as_ubyte(im_all)

			#im2 = Image.fromarray(im).convert('RGB')

			#video.write(im)

			#out.write(im)


			if cv2.waitKey(1) & 0xFF == ord('s'):
				break
			time.sleep(0.5)
		#---------------------------------------------------------
		# 
		#clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(im_all, fps=1)
		#clip.write_videofile('my_video.mp4')

		#video.write(im_all)

		#imageio.mimwrite('output_filename.mp4', im_all, fps = 1)
		path2 = os.path.join(loc,folder,"output_filename.mp4")
		imageio.mimsave(path2, im_all, fps = 1)
