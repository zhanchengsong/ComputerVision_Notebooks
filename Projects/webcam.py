import copy
import cv2
import matplotlib.pyplot as plt

# Set the default figure size
plt.rcParams['figure.figsize'] = [20,10]
# Load the training Image and convert to grayscale
image = cv2.imread('./faces/test.jpg')
training_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
training_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Display the images
plt.subplot(121)
plt.title('Original Training Image')
plt.imshow(training_image)
plt.subplot(122)
plt.title('Gray Scale Training Image')
plt.imshow(training_gray, cmap='gray')
plt.show()

# Locating key points using ORB
plt.rcParams['figure.figsize'] = [14.0, 7.0]
# Set the parameters of the ORB algorithm by specifying the maximum number of keypoints to locate and
# the pyramid decimation ratio
orb = cv2.ORB_create(1000, 2.0)
# Find the keypoints in the gray scale training image and compute their ORB descriptor.
# The None parameter is needed to indicate that we are not using a mask.
keypoints, descriptor = orb.detectAndCompute(training_gray, None)

# Create copies of the training image to draw our keypoints on
keyp_without_size = copy.copy(training_image)
keyp_with_size = copy.copy(training_image)

# Draw the keypoints without size or orientation on one copy of the training image
cv2.drawKeypoints(training_image, keypoints, keyp_without_size, color = (0, 255, 0))

# Draw the keypoints with size and orientation on the other copy of the training image
cv2.drawKeypoints(training_image, keypoints, keyp_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with the keypoints without size or orientation
plt.subplot(121)
plt.title('Keypoints Without Size or Orientation')
plt.imshow(keyp_without_size)

# Display the image with the keypoints with size and orientation
plt.subplot(122)
plt.title('Keypoints With Size and Orientation')
plt.imshow(keyp_with_size)
plt.show()

# Print the number of keypoints detected
print("\nNumber of keypoints Detected: ", len(keypoints))
# Get the training keypoints
keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)

#Matching keypoints against the camera image
cam = cv2.VideoCapture(0)
while 1 == 1:
    ret_val, query_image = cam.read()
    # Convert the query image to gray scale
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)
    # Create a Brute Force Matcher object. Set crossCheck to True so that the BFMatcher will only return consistent
    # pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Perform the matching between the ORB descriptors of the training image and the query image
    matches = bf.match(descriptors_train, descriptors_query)

    # Connect the keypoints in the training image with their best matching keypoints in the query image.
    # The best matches correspond to the first elements in the sorted matches list, since they are the ones
    # with the shorter distance. We draw the first 300 mathces and use flags = 2 to plot the matching keypoints
    # without size or orientation.
    result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:300], query_image,
                             flags=2)

    cv2.imshow('Result',result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
