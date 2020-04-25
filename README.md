# Image-Morph
In the same folder, keep the source image, destination image and the python file. Make sure you have python 3 installed. Then follow the below steps:

 1. (Optional step) In the terminal type `pip install opencv-python` if you dont have opencv(cv2) installed.
 2. Run the python file in the terminal (console) by typing `python
    image_morph.py`
 3. Enter the Source file name with file extension in the terminal when prompted. *eg- Clinton.jpg*
 4. Use your mouse to click on the desired control points. You can select up-to 100 control points (as our Delaunay algorithm works fine for less control points only). The coordinated shall get displayed on the terminal
 5. Press any key to view the triangulated image.
 6. Repeat steps 2,3,4 for the destination image.
 7. Enter the step size (1/number of desired frames) when prompted. *eg- Enter 0.2 if you want 5 frames*.
 8. The intermediary frames shall be saved in the same folder
