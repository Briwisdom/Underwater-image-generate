 This is a Graduation Design for Undergraduates.
# Underwater-image-generate

 In this experimence, We have used four type GAN：DCGAN，WGAN_GP，ACGAN，SGAN running this code, if you want to run this code, just choose the version of GAN, then change some code about the file path of train dataset. For example:

python dcgan.py

The underwater image data collection from web crawler collection, there are five type: Coral, Jellyfish, Parrotfish, SeaAnemone, Zebrafish. 
The follows images are acgan(five classes), sgan(five classes), dcgan(Parrotfish),wgan_gp(Parrotfish), original images(Parrotfish), respectively:

![acgan](https://github.com/Briwisdom/Underwater-image-generate/blob/master/acgan/images/1100.png)

![sgan](https://github.com/Briwisdom/Underwater-image-generate/blob/master/sgan/images_64/mnist_1150.png)

![dcgan,Parrotfish](https://github.com/Briwisdom/Underwater-image-generate/blob/master/dcgan/genImages/Parrotfish_64/epochs_1120.png)

![wgan_gp, (Parrotfish)](https://github.com/Briwisdom/Underwater-image-generate/blob/master/wgan_gp/genImages/Parrotfish_64/epochs_1620.png)

![original images, Parrotfish](https://github.com/Briwisdom/Underwater-image-generate/blob/master/Parrotfish_64.png)

the dataset and some generate images can be find in the in the path:

