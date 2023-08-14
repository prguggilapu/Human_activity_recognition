# Human_activity_recognition

This model is to build a model to predict human activities such as<br />
1 WALKING<br />
2 WALKING_UPSTAIRS<br />
3 WALKING_DOWNSTAIRS<br />
4 SITTING<br />
5 STANDING<br />
6 LAYING<br />

# Train and Test 
The dataset is randomly partitioned into two sets, where 70% of the volunteers were selected for generating the train data and 30% for the test data. The experiments are carried with 30 volunteers. 

# How the data was recorded
The inertial signals are the raw signals from the accelerometer and gyroscope in a smartphone. The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). 

# Feature Selection 
The features selected for this database come from the accelerometer and gyroscope 3-axial raw signals tAcc-XYZ and tGyro-XYZ. These time domain signals (prefix 't' to denote time) were captured at a constant rate of 50 Hz. Then they were filtered using a median filter and a 3rd order low pass Butterworth filter with a corner frequency of 20 Hz to remove noise. Similarly, the acceleration signal was then separated into body and gravity acceleration signals (tBodyAcc-XYZ and tGravityAcc-XYZ) using another low pass Butterworth filter with a corner frequency of 0.3 Hz.<br />

Subsequently, the body linear acceleration and angular velocity were derived in time to obtain Jerk signals (tBodyAccJerk-XYZ and tBodyGyroJerk-XYZ). Also the magnitude of these three-dimensional signals were calculated using the Euclidean norm (tBodyAccMag, tGravityAccMag, tBodyAccJerkMag, tBodyGyroMag, tBodyGyroJerkMag).<br />

Finally a Fast Fourier Transform (FFT) was applied to some of these signals producing fBodyAcc-XYZ, fBodyAccJerk-XYZ, fBodyGyro-XYZ, fBodyAccJerkMag, fBodyGyroMag, fBodyGyroJerkMag. (Note the 'f' to indicate frequency domain signals).<br />

These signals were used to estimate variables of the feature vector for each pattern:  
'-XYZ' is used to denote 3-axial signals in the X, Y and Z directions.<br />

tBodyAcc-XYZ<br />
tGravityAcc-XYZ<br />
tBodyAccJerk-XYZ<br />
tBodyGyro-XYZ<br />
tBodyGyroJerk-XYZ<br />
tBodyAccMag<br />
tGravityAccMag<br />
tBodyAccJerkMag<br />
tBodyGyroMag<br />
tBodyGyroJerkMag<br />
fBodyAcc-XYZ<br />
fBodyAccJerk-XYZ<br />
fBodyGyro-XYZ<br />
fBodyAccMag<br />
fBodyAccJerkMag<br />
fBodyGyroMag<br />
fBodyGyroJerkMag<br />

The set of variables that were estimated from these signals are: <br />

mean(): Mean value<br />
std(): Standard deviation<br />
mad(): Median absolute deviation<br /> 
max(): Largest value in array<br />
min(): Smallest value in array<br />
sma(): Signal magnitude area<br />
energy(): Energy measure. Sum of the squares divided by the number of values.<br /> 
iqr(): Interquartile range <br />
entropy(): Signal entropy<br />
arCoeff(): Autorregresion coefficients with Burg order equal to 4<br />
correlation(): correlation coefficient between two signals<br />
maxInds(): index of the frequency component with largest magnitude<br />
meanFreq(): Weighted average of the frequency components to obtain a mean frequency<br />
skewness(): skewness of the frequency domain signal <br />
kurtosis(): kurtosis of the frequency domain signal <br />
bandsEnergy(): Energy of a frequency interval within the 64 bins of the FFT of each window.<br />
angle(): Angle between to vectors.<br />

Additional vectors obtained by averaging the signals in a signal window sample. These are used on the angle() variable:<br />

gravityMean <br />
tBodyAccMean<br />
tBodyAccJerkMean<br />
tBodyGyroMean<br />
tBodyGyroJerkMean<br />

# Algorithms
For raw data we will be using RNN models and for expert-engineered features we use the classic ML algorithms<br />
