Patch Matching Program:

main.py: Python Executing file.

********Demo*********
Executing the file main enable to test the program with the desired model on the chosen image and patch.

******Before Running********

Define the parameters with the image and the patch files, the name for saving, the number of patch to be found and the zoom parameters.

Chose the estimation method: 
- patch_matching_NB: could be very long because of the naive covariance matrix computation 
- patch_matching_color: 3 features retained (R,G,B)
- patch_maching_color_5parameters:5 features retained (R,G,B,x,y)
- patch_matching_color_squareparameters:14 features retained (x,y,R_1,G_1,B_1,R_2,G_2,B_2,R_3,G_3,B_3,R_4,G_4,B_4)

********Changing parameters******
The method dispresult displays the image with white boxes for the patch found. Once the estimation method has been runned, you can run again and again the KLmin and the dispresult methods while changing the parameters, mainly the patch_number parameters and except the zoom parameters, without having to run all the file.

