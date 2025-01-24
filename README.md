# Cell Detection and Tracking System

## Overview
This system is designed for automated detection and tracking of cells and cell clusters in video microscopy data, with a specific focus on analyzing cell movement patterns around organoids. The system processes video frames to identify, track, and analyze the movement of individual cells and cell clusters over time. The system comes in two versions:

### 1.  Single Organoid System (Main_Tracking_OneOrganoid)
- Optimized for tracking cell movements around a single organoid
- Simplified zone classification (near vs. far zones)

### 2. Two Organoids System (Main_Tracking_TwoOrganoids)
- Enhanced tracking for environments with two organoids
- Complex zone classification (near zones for each organoid, between-organoid zone, far zones)


## Key Features

### 1. Object Detection and Classification
- Uses adaptive thresholding with dynamic parameters
- Implements circularity checks for improved cell identification
- Breaks bridges between connected objects using morphological operations
- Provides continuous tracking across video frames

### 2. Advanced Tracking Capabilities
- Maintains object persistence across frames
- Implements predictive tracking using velocity calculations
- Uses frame averaging to reduce noise and improve detection stability
- Handles object occlusion and reappearance
- Tracks objects through multiple time periods

### 3. Movement Analysis
- Grid-based spatial analysis of cell movements
- Calculation of movement vectors and trajectories
- Analysis of movement patterns relative to organoid positions
- Angle calculations between movement vectors and organoid centers
- Classification of movements into near and far zones

### 4. Visualization Features
- Real-time tracking visualization with object highlighting
- Trajectory visualization showing cell movement paths
- Grid-based movement pattern visualization
- Angle-based movement analysis visualization
- Color-coded zone classification

### 5. Data Analysis and Statistics
- Comprehensive tracking statistics
- Period-based analysis of cell movements
- Track continuity analysis across period boundaries
- Movement pattern histograms
- Grid-based spatial distribution analysis

## Processing Steps

### 1. System Initialization
1. Configuration Setup:
   - Loading system parameters (ROI padding, thresholds, tracking settings)
   - Setting up video processing parameters (FPS, duration, slow motion factor)
   - Initializing detection and tracking parameters
   - Setting up grid configuration for spatial analysis

2. ROI (Region of Interest) Setup:
   - Reading video dimensions
   - Calculating ROI coordinates with padding
   - Validating ROI dimensions against video size
   - Setting up coordinate system for tracking

3. Output Directory Creation:
   - Creating timestamped output directory
   - Setting up file paths for various outputs
   - Initializing video writers for different visualizations
   - Preparing data storage structures

4. Organoid Data Loading:
   - Reading organoid tracking data from JSON
   - Validating data format and structure
   - Setting up interpolation parameters
   - Initializing period-based analysis structures

### 2. Frame Processing Pipeline
1. Frame Preprocessing:
   - Color space conversion (BGR to Grayscale)
   - Median blur application for noise reduction
   - Contrast stretching:
     * Clipping intensity values
     * Normalizing intensity range
     * Enhancing image contrast
   - Adaptive thresholding:
     * Creating threshold masks for tracked objects
     * Applying different threshold levels for known cell locations
     * Cleaning up thresholded image with morphological operations

2. Object Detection:
   - Bridge breaking between connected objects:
     * Skeletonization of binary image
     * Distance transform calculation
     * Identification and removal of narrow connections
   - Contour detection and analysis:
     * Finding external contours
     * Calculating contour areas and perimeters
     * Computing circularity metrics
   - Object classification:
     * Differentiating between single cells and clusters
     * Applying size and shape thresholds
     * Handling multi-cell clusters
   - Position calculation:
     * Computing centroids using moments
     * Converting to global coordinates
     * Storing contour information

3. Object Tracking:
   - Frame buffering:
     * Maintaining rolling buffer of recent frames
     * Averaging detections across multiple frames
     * Reducing noise and false detections
   - Track matching:
     * Predicting object positions
     * Calculating movement vectors
     * Matching detections with existing tracks
   - Track maintenance:
     * Updating position histories
     * Managing track lifecycles
     * Handling lost and found objects
   - Position smoothing:
     * Calculating weighted averages
     * Applying velocity-based predictions
     * Managing track continuity

### 3. Analysis Components
1. Spatial Analysis:
   - Grid-based zone classification:
     * Dividing ROI into grid cells
     * Classifying zones based on organoid proximity
     * Calculating dynamic zone thresholds
   - Movement vector analysis:
     * Computing movement directions
     * Calculating vector magnitudes
     * Analyzing movement patterns
   - Angle analysis:
     * Computing angles relative to organoid centers
     * Creating angular distribution histograms
     * Analyzing directional preferences

2. Statistical Analysis:
   - Track statistics:
     * Computing track lengths and durations
     * Analyzing track continuity
     * Calculating detection frequencies
   - Period analysis:
     * Analyzing tracks across period boundaries
     * Computing period-based statistics
     * Tracking object persistence
   - Movement pattern analysis:
     * Analyzing velocity distributions
     * Computing spatial distribution patterns
     * Generating movement histograms

3. Temporal Analysis:
   - Period boundary processing:
     * Identifying period transitions
     * Managing track continuity
     * Computing period-based metrics
   - Track continuity analysis:
     * Analyzing track splits and merges
     * Computing track persistence
     * Identifying track relationships

### 4. Output Generation
1. Video Outputs:
   - Main tracking visualization:
     * Drawing tracked objects
     * Visualizing organoid positions
     * Adding frame information
   - Trajectory visualization:
     * Plotting movement paths
     * Color-coding trajectories
     * Showing temporal progression
   - Grid-based analysis:
     * Visualizing spatial zones
     * Showing movement vectors
     * Displaying zone statistics
   - Angle-based visualization:
     * Color-coding movement angles
     * Showing directional patterns
     * Displaying angle distributions

2. Data Outputs:
   - Statistical reports:
     * Tracking statistics in CSV format
     * Period analysis reports
     * Track continuity analysis
   - Analysis visualizations:
     * Movement histograms
     * Angle distribution plots
     * Zone distribution graphs
   - Analysis data:
     * JSON formatted tracking data
     * Period-based statistics
     * Aggregate analysis results

3. Quality Control:
   - Validation checks:
     * Verifying track consistency
     * Checking data completeness
     * Validating analysis results
   - Error handling:
     * Managing missing frames
     * Handling detection failures
     * Recording processing issues

## Configuration Parameters
The system includes a comprehensive configuration class (`Config`) with adjustable parameters for:
- ROI settings
- Video processing parameters
- Detection thresholds
- Tracking parameters
- Analysis grid settings
- Visualization parameters

## Output Files
The system generates several output files:
1. Tracked videos:
   - Main tracking visualization
   - Trajectory visualization
   - Grid-based analysis
   - Angle-based analysis

2. Analysis files:
   - Tracking statistics (CSV)
   - Period analysis (TXT)
   - Track continuity analysis (TXT)
   - Movement histograms (PNG)
   - Aggregate analysis data (JSON)

## Technical Requirements
- OpenCV (cv2)
- NumPy
- Matplotlib
- SciPy
- Scikit-image
- Pandas

## Usage

### Single Organoid System
```python
# For single organoid analysis
from Main_Tracking_OneOrganoid import CellTracker
tracker = CellTracker(video_path)
output_path = tracker.process_video()
```

### Two Organoids System
```python
# For two organoids analysis
from Main_Tracking_TwoOrganoids import CellTracker
tracker = CellTracker(video_path)
output_path = tracker.process_video()
```

Both versions follow the same usage pattern but provide different analysis outputs based on their specialized configurations.