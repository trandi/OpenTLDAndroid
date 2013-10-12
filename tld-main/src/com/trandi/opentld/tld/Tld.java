/**
 * Copyright 2013 Dan Oprescu
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.trandi.opentld.tld;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.util.Log;

import com.trandi.opentld.tld.Parameters.ParamsTld;
import com.trandi.opentld.tld.Util.DefaultRNG;
import com.trandi.opentld.tld.Util.NNConfStruct;
import com.trandi.opentld.tld.Util.Pair;
import com.trandi.opentld.tld.Util.RNG;


public class Tld {
	private static final int MAX_DETECTED = 100;
	
	
	ParamsTld _params;
	FernEnsembleClassifier _classifierFern;
	NNClassifier _classifierNN;
	private final LKTracker _tracker = new LKTracker();
	private PatchGenerator _patchGenerator; // FIXME UNUSED, why !?
	private final RNG _rng = new DefaultRNG();
	
	
	// Integral Images
	private int _iiRows;
	private int _iiCols;
	private final Mat _iisum = new Mat();
	private final Mat _iisqsum = new Mat();
	// for performance reasons, duplicate the data directly in Java, to avoid too many native code invocations
	private int[] _iisumJava;
	private double[] _iisqsumJava;
	private float _var; // variance of the initial patch/box. Will be used by the 1st stage of the classifier.
	
	// Training data
	Mat _pExample = new Mat(); // positive NN example
	final List<Pair<int[], Boolean>> _pFerns = new ArrayList<Pair<int[], Boolean>>();	//positive ferns <allFernsHashCodes, true>
	private final List<Mat> _pPatterns = new ArrayList<Mat>(); //positive patches to display
	private List<Mat> _nExamples;

	// Last frame data
	private BoundingBox _lastbox;
	private boolean _learn = true;
	
	// Detector data
	private Map<BoundingBox, int[]> _fernDetectionNegDataForLearning = new HashMap<BoundingBox, int[]>(); // all ferns hash codes for a given bounding box
	final Map<DetectionStruct, Integer> _boxClusterMap = new HashMap<DetectionStruct, Integer>();	// the cluster to which each detected box belongs
	
	// Bounding Boxes Grid
	Grid _grid;
	  
	
	public Tld(Properties parameters){
		_params = new ParamsTld(parameters);
		_classifierFern = new FernEnsembleClassifier(parameters);
		_classifierNN = new NNClassifier(parameters);
		_patchGenerator = new PatchGenerator(0, 0, _params.noise_init, true, 1 - _params.scale_init, 1 + _params.scale_init,
				-_params.angle_init * Math.PI / 180f, _params.angle_init * Math.PI / 180f, 
				-_params.angle_init * Math.PI / 180f, _params.angle_init * Math.PI / 180f);
	
		_pExample.create(_params.patch_size, _params.patch_size, CvType.CV_64F);
	}

	protected Tld() {
		// for TESTING only
	}

	public void init(Mat frame1, Rect trackedBox) {
		// get Bounding boxes
		if(Math.min(trackedBox.width, trackedBox.height) < _params.min_win) {
			throw new IllegalArgumentException("Provided trackedBox: " + trackedBox + " is too small (min " + _params.min_win + ")");
		}
		_grid = new Grid(frame1, trackedBox, _params.min_win);
		Log.i(Util.TAG, "Init Created " + _grid.getSize() + " bounding boxes.");
		_grid.updateGoodBadBoxes(trackedBox, _params.num_closest_init);
		
		_iiRows = frame1.rows();
		_iiCols = frame1.cols();
		_iisum.create(_iiRows, _iiCols, CvType.CV_32F);
		_iisqsum.create(_iiRows, _iiCols, CvType.CV_64F);
		
		// correct bounding box
		_lastbox = _grid.getBestBox();
		
		_classifierFern.init(_grid.getTrackedBoxScales(), _rng);
		
		
		// generate DATA
		// generate POSITIVE DATA
		generatePositiveData(frame1, _params.num_warps_init, _grid);
		
		// Set variance threshold
		MatOfDouble stddev = new MatOfDouble();
		Core.meanStdDev(frame1.submat(_grid.getBestBox()), new MatOfDouble(), stddev);
		updateIntegralImgs(frame1);
		// this is directly half of the variance of the initial box, which will be used the the 1st stage of the classifier
		_var = (float)Math.pow(stddev.toArray()[0], 2d) * 0.5f;
		// check variance
		final double checkVar = Util.getVar(_grid.getBestBox(), _iisumJava, _iisqsumJava, _iiCols) * 0.5;
		Log.i(Util.TAG, "Variance: " + _var + " / Check variance: " + checkVar);
		
		
		// generate NEGATIVE DATA
		final Pair<List<Pair<int[], Boolean>>, List<Mat>> negData = generateNegativeData(frame1);
		
		// Split Negative Ferns <features, labels=false> into Training and Testing sets (they are already shuffled)
		final int nFernsSize = negData.first.size();
		final List<Pair<int[], Boolean>> nFernsTest = new ArrayList<Pair<int[], Boolean>>(negData.first.subList(0, nFernsSize/2));
		final List<Pair<int[], Boolean>> nFerns = new ArrayList<Pair<int[], Boolean>>(negData.first.subList(nFernsSize/2, nFernsSize));
		
		// Split Negative NN Examples into Training and Testing sets
		final int nExSize = negData.second.size();
		final List<Mat> nExamplesTest = new ArrayList<Mat>(negData.second.subList(0, nExSize/2));
		_nExamples = new ArrayList<Mat>(negData.second.subList(nExSize/2, nExSize));

		
		//MERGE Negative Data with Positive Data and shuffle it
		final List<Pair<int[], Boolean>> fernsData = new ArrayList<Pair<int[], Boolean>>(_pFerns);
		fernsData.addAll(nFerns);
		Collections.shuffle(fernsData);
		
		// TRAINING
		Log.i(Util.TAG, "Init Start Training with " + fernsData.size() + " ferns, " 
		+ _nExamples.size() + " nExamples, " + nFernsTest.size() + " nFernsTest, " + nExamplesTest.size() + " nExamplesTest");
		_classifierFern.trainF(fernsData, 10);
		_classifierNN.trainNN(_pExample, _nExamples);
		// Threshold evaluation on testing sets
		_classifierFern.evaluateThreshold(nFernsTest);
		_classifierNN.evaluateThreshold(nExamplesTest);
	}

	private void updateIntegralImgs(Mat frame) {
		Imgproc.integral2(frame, _iisum, _iisqsum);
		// duplicate the data for performance reasons
		_iisumJava = Arrays.copyOf(Util.getIntArray(_iisum), _iiRows * _iiCols);
		_iisqsumJava = Arrays.copyOf(Util.getDoubleArray(_iisqsum), _iiRows * _iiCols);
	}
	
	public ProcessFrameStruct processFrame(final Mat lastImg, final Mat currentImg){
		// 1. TRACK
		TrackingStruct trackingStruct = null;
		if(_lastbox != null){
			trackingStruct = track(lastImg, currentImg, _lastbox);
		}
		
			
		
		// 2. DETECT
		final Pair<List<DetectionStruct>, List<DetectionStruct>> detStructs = detect(currentImg);
		
		// 3. INTEGRATION tracking with detection
		if(trackingStruct != null){
			_lastbox = trackingStruct.predictedBB;
			if(trackingStruct.conf > _classifierNN.getNNThresholdValid()){
				Log.i(Util.TAG, "Tracking confidence: " + trackingStruct.conf + " > " + " Threshold: " + _classifierNN.getNNThresholdValid() + " ===> WILL LEARN");
				_learn = true;
			}else{
				Log.i(Util.TAG, "Tracking confidence: " + trackingStruct.conf + " < " + " Threshold: " + _classifierNN.getNNThresholdValid() + " ===> WILL NOT LEARN");
			}
			
			Log.i(Util.TAG, "Tracked");
			if(detStructs != null){
				final Map<BoundingBox, Float> clusters = clusterConfidentIndices(detStructs.second);// cluster detections
				Log.i(Util.TAG, "Found " + clusters.size() + " clusters");
				final Map<BoundingBox, Float> confidentClusters = new HashMap<BoundingBox, Float>();
				for(BoundingBox clusterBox : clusters.keySet()){
					// Get clusters that are far from tracker and with better confidence
					if(trackingStruct.predictedBB.calcOverlap(clusterBox) < 0.5 && clusters.get(clusterBox) > trackingStruct.conf){
						confidentClusters.put(clusterBox, clusters.get(clusterBox));
					}
				}
				
				if(confidentClusters.size() == 0){
					Log.i(Util.TAG, "NO NN confident cluster !");
				}else if(confidentClusters.size() == 1){
					Log.i(Util.TAG, "Detected better match (1 confident cluster), re-initialising tracker");
					_lastbox = confidentClusters.keySet().iterator().next(); //bbnext
					_learn = false;
				}else{
					Log.i(Util.TAG, "Plenty of confident clusters detected. Get mean of close detections (use nnMatches)");
					int cx=0,cy=0,cw=0,ch=0, close_detections=0;
					for(DetectionStruct detStruct : detStructs.second){
						if(trackingStruct.predictedBB.calcOverlap(detStruct.detectedBB) > 0.7){
							cx += detStruct.detectedBB.x;
							cy += detStruct.detectedBB.y;
							cw += detStruct.detectedBB.width;
							ch += detStruct.detectedBB.height;
							close_detections++;
						}
					}
					
					if(close_detections > 0){
						// weighted average (10 to 1 in favour of the tracked) trackers trajectory with the close detections
						_lastbox.x = Math.round((float)(10*trackingStruct.predictedBB.x+cx)/(float)(10+close_detections));
						_lastbox.y = Math.round((float)(10*trackingStruct.predictedBB.y+cy)/(float)(10+close_detections));
						_lastbox.width = Math.round((float)(10*trackingStruct.predictedBB.width+cw)/(float)(10+close_detections));
						_lastbox.height =  Math.round((float)(10*trackingStruct.predictedBB.height+ch)/(float)(10+close_detections));
					}
				}
			}
		}else{ // IF NOT Tracking
			Log.w(Util.TAG, "NOT Tracking");
			_lastbox = null;
			_learn = false;
			if(detStructs != null){  // and detector is defined
				final Map<BoundingBox, Float> clusters = clusterConfidentIndices(detStructs.second);// cluster detections
				if(clusters.size() == 1){
					// not tracking but detected exactly 1 cluster -> use this one as the best option
					_lastbox = clusters.keySet().iterator().next();
				}
			}
		}
		
		
		// 4. LEARN
		if(_learn){
			_learn = learn(currentImg, detStructs != null ? detStructs.first : null); // use the Fern classifier detected
		}else{
			Log.i(Util.TAG, "NOT Learning");
		}

		
		
		
		final Point[] lastPoints = (trackingStruct == null ? null : trackingStruct.lastPoints);
		final Point[] currentPoints = (trackingStruct == null ? null : trackingStruct.currentPoints);
		return new ProcessFrameStruct(lastPoints, currentPoints, _lastbox);
	}
	
	
	
	
	
	private TrackingStruct track(final Mat lastImg, final Mat currentImg, final BoundingBox lastBox) {
		Log.i(Util.TAG, "[TRACK]");
		
		// Generate points
		final Point[] lastPoints = lastBox.points();
		if(lastPoints.length == 0){
			Log.e(Util.TAG, "Points not generated from lastBox: " + lastBox);
			return null;
		}
		
		
		// Frame-to-frame tracking with forward-backward error checking
		final Pair<Point[], Point[]> trackedPoints = _tracker.track(lastImg, currentImg, lastPoints);
		if(trackedPoints == null){
			Log.e(Util.TAG, "No points could be tracked.");
			return null;			
		}
		if(_tracker.getMedianErrFB() > _params.tracker_stability_FBerrMax){
			Log.w(Util.TAG, "TRACKER too unstable. FB Median error: " + _tracker.getMedianErrFB() + " > " + _params.tracker_stability_FBerrMax);
			// return null;  // we hope the detection will find the pattern again
		}
		
		// bounding box prediction
		final BoundingBox predictedBB = lastBox.predict(trackedPoints.first, trackedPoints.second);
		if(predictedBB.x > currentImg.cols() || predictedBB.y > currentImg.rows()
				|| predictedBB.br().x < 1 || predictedBB.br().y < 1)
		{
			Log.e(Util.TAG, "TRACKER Predicted bounding box out of range !");
			return null;
		}

		// estimate Confidence
		Mat pattern = new Mat();
		try{
			resizeZeroMeanStdev(currentImg.submat(predictedBB.intersect(currentImg)), pattern, _params.patch_size);
		}catch(Throwable t){
			Log.e(Util.TAG, "PredBB when failed: " + predictedBB);
		}
		//Log.i(Util.TAG, "Confidence " + pattern.dump());		
		
		//Conservative Similarity
		final NNConfStruct nnConf = _classifierNN.nnConf(pattern);
		Log.i(Util.TAG, "Tracking confidence: " + nnConf.conservativeSimilarity);
		
		Log.i(Util.TAG, "[TRACK END]");
		return new TrackingStruct(nnConf.conservativeSimilarity, predictedBB, trackedPoints.first, trackedPoints.second);
	}
	
	
	/**
	 * Structure the classifier into 3 stages:
	 * a) patch variance
	 * b) ensemble of ferns classifier
	 * c) nearest neighbour
	 */
	private Pair<List<DetectionStruct>, List<DetectionStruct>> detect(final Mat frame){
		Log.i(Util.TAG, "[DETECT]");
		
		final List<DetectionStruct> fernClassDetected = new ArrayList<Tld.DetectionStruct>(); //dt
		final List<DetectionStruct> nnMatches = new ArrayList<Tld.DetectionStruct>(); //dbb
		
		
		// 0. Cleaning
		_boxClusterMap.clear();
		
		// 1. DETECTION
		final Mat img = new Mat(frame.rows(), frame.cols(), CvType.CV_8U);
		updateIntegralImgs(frame);
		Imgproc.GaussianBlur(frame, img, new Size(9, 9), 1.5);
		
		// Apply the Variance filter TODO : Bottleneck
		int a=0;
		for(BoundingBox box : _grid){
			// a) speed up by doing the features/ferns check ONLY if the variance is high enough !
			if(Util.getVar(box, _iisumJava, _iisqsumJava, _iiCols) >= _var ){
				a++;
				final Mat patch = img.submat(box);
				final int[] allFernsHashCodes = _classifierFern.getAllFernsHashCodes(patch, box.scaleIdx);
				final double averagePosterior = _classifierFern.averagePosterior(allFernsHashCodes);
				_fernDetectionNegDataForLearning.put(box,  allFernsHashCodes);// store for later use in learning
				
				// b)
				if(averagePosterior > _classifierFern.getFernPosThreshold()){
					fernClassDetected.add(new DetectionStruct(box, allFernsHashCodes, averagePosterior, patch));
				}
			}
		}
		
		Log.i(Util.TAG, a + " Bounding boxes passed the variance filter (" + _var + ")");
		Log.i(Util.TAG, fernClassDetected.size() + " Initial detected from Fern Classifier");
		if(fernClassDetected.size() == 0){
			Log.i(Util.TAG, "[DETECT END]");
			return null;
		}
		
		// keep only the best
		Util.keepBestN(fernClassDetected, MAX_DETECTED, new Comparator<DetectionStruct>() {
			@Override
			public int compare(DetectionStruct detS1, DetectionStruct detS2) {
				return Double.compare(detS1.averagePosterior, detS2.averagePosterior);
			}
		});
		
		
		// 2. MATCHING using the NN classifier  c)
		for(DetectionStruct detStruct : fernClassDetected){
			// update detStruct.patch to params.patch_size and normalise it
			Mat pattern = new Mat();
			resizeZeroMeanStdev(detStruct.patch, pattern, _params.patch_size);
			detStruct.nnConf = _classifierNN.nnConf(pattern);
			
			Log.i(Util.TAG, "NNConf: " + detStruct.nnConf.relativeSimilarity + " / " + detStruct.nnConf.conservativeSimilarity + " Threshold: " + _classifierNN.getNNThreshold());
			// only keep valid boxes
			if(detStruct.nnConf.relativeSimilarity > _classifierNN.getNNThreshold()){
				nnMatches.add(detStruct); 
			}
		}
		
		Log.i(Util.TAG, "[DETECT END]");
		return new Pair<List<DetectionStruct>, List<DetectionStruct>>(fernClassDetected, nnMatches);
	}
	
	
	private boolean learn(final Mat img, final List<DetectionStruct> fernClassDetected){
		Log.i(Util.TAG, "[LEARN]");
		Mat pattern = new Mat();
		final double stdev = resizeZeroMeanStdev(img.submat(_lastbox.intersect(img)), pattern, _params.patch_size);
		final NNConfStruct confStruct = _classifierNN.nnConf(pattern);
		
		if(confStruct.relativeSimilarity < 0.5){
			Log.w(Util.TAG, "Fast change, NOT learning");
			return false;
		}
		if(Math.pow(stdev, 2) < _var){
			Log.w(Util.TAG, "Low variance, NOT learning");
			return false;
		}
		if(confStruct.isin.inNegSet){
			Log.w(Util.TAG, "Patch in negative data, NOT learning");
			return false;
		}
		
		// Data generation
		_grid.updateGoodBadBoxes(_lastbox, _params.num_closest_update);
		if(_grid.getGoodBoxes().length > 0){
			generatePositiveData(img,  _params.num_warps_update, _grid);
		}else{
			Log.w(Util.TAG, "NO good boxes, NOT learning.");
			return false;
		}
		
		// TODO why don't we learn from the GOOD boxes too !?
		final List<Pair<int[], Boolean>> fernExamples = new ArrayList<Util.Pair<int[], Boolean>>(_pFerns);
		for(BoundingBox badBox : _grid.getBadBoxes()){
			final int[] allFernsHashCodes = _fernDetectionNegDataForLearning.get(badBox);
			if(allFernsHashCodes != null){
				// these are NEGATIVE examples !
				fernExamples.add(new Pair<int[], Boolean>(allFernsHashCodes, false));
			}
		}
		
		final List<Mat> nnExamples = new ArrayList<Mat>();
		if(fernClassDetected != null){
			for(DetectionStruct detStruct : fernClassDetected){
				if(_lastbox.calcOverlap(detStruct.detectedBB) < Grid.BAD_OVERLAP){
					nnExamples.add(detStruct.patch);
				}
			}
		}
		
		// Classifiers update
		_classifierFern.trainF(fernExamples, 2);
		_classifierNN.trainNN(_pExample, _nExamples);
		
		Log.i(Util.TAG, "[LEARN END]");
		return true;
	}
	
	
	
	/**
	 * 
	 * @param conservativeSimilarities
	 * @return Map of clusters' boxes and their confidence
	 */
	private Map<BoundingBox, Float> clusterConfidentIndices(final List<DetectionStruct> conservativeSimilarities){
		final int numbb = conservativeSimilarities.size();
		if(numbb == 0){
			Log.i(Util.TAG, "NO conservative similarities provided, NOTHING to cluster.");
			return new HashMap<BoundingBox, Float>(); // empty result
		}
		
		
		// by default there is only 1 cluster, and ALL boxes are in it (0)
		int clusters = 1;
		for(DetectionStruct detStruct : conservativeSimilarities){
			_boxClusterMap.put(detStruct, 0);
		}
		
		
		if(numbb == 1){
			return Collections.singletonMap(conservativeSimilarities.get(0).detectedBB, conservativeSimilarities.get(0).nnConf.conservativeSimilarity);
		}else if(numbb == 2){
			if(conservativeSimilarities.get(0).detectedBB.calcOverlap(conservativeSimilarities.get(1).detectedBB) < 0.5){
				// 2nd box is in its own cluster, update
				_boxClusterMap.put(conservativeSimilarities.get(1), 1);
				clusters = 2;
			}
		}else {
			clusters = clusterBB();
		}
		
		final Map<BoundingBox, Float> result = new HashMap<BoundingBox, Float>();
		
		for(int cluster = 0; cluster < clusters; cluster++){
			float avgConservativeSimilarity = 0f;
			int clusterBoxCount = 0, mx=0, my=0, mw=0, mh=0;
			
			for(DetectionStruct detStruct : _boxClusterMap.keySet()){
				if(_boxClusterMap.get(detStruct) == cluster){
					avgConservativeSimilarity += detStruct.nnConf.conservativeSimilarity;
					mx += detStruct.detectedBB.x;
					my += detStruct.detectedBB.y;
					mw += detStruct.detectedBB.width;
					mh += detStruct.detectedBB.height;
					clusterBoxCount++;
				}
			}
			
			if(clusterBoxCount > 0){
				final BoundingBox clusterBox = new BoundingBox();
				clusterBox.x = mx / clusterBoxCount;
				clusterBox.y = my / clusterBoxCount;
				clusterBox.width = mw / clusterBoxCount;
				clusterBox.height = mh / clusterBoxCount;
				
				result.put(clusterBox, avgConservativeSimilarity / clusterBoxCount);
			}
		}
		
		return result;
	}
	
	
	/**
	 * @param boxClusterMap INPUT / OUTPUT
	 * @return Total clusters count
	 */
	private int clusterBB(){
		final int size = _boxClusterMap.size();
		// need the data in arrays
		final DetectionStruct[] dbb = _boxClusterMap.keySet().toArray(new DetectionStruct[size]);
		final int[] indexes = new int[size];
		for(int i = 0; i < size; i++){
			indexes[i] = _boxClusterMap.get(dbb[i]);
		}
		
		// 1. Build proximity matrix
		final float[] data = new float[size * size];
		for(int i = 0; i < size; i++){
			for(int j = 0; j < size; j++){
				final float d = 1 - dbb[i].detectedBB.calcOverlap(dbb[j].detectedBB);
				data[i * size + j] = d;
				data[j * size + i] = d;
			}
		}
		Mat D = new Mat(size, size, CvType.CV_32F);
		D.put(0, 0, data);
		
		// 2. Initialise disjoint clustering
		final int[] belongs = new int[size];
		int m = size;
		for(int i = 0; i < size; i++){
			belongs[i] = i;
		}
		

		for(int it = 0; it < size - 1; it++){
			//3. Find nearest neighbour
			float min_d = 1;
			int node_a = -1, node_b = -1;
			for (int i = 0; i < D.rows(); i++){
				for (int j = i + 1 ;j < D.cols(); j++){
					if (data[i * size + j] < min_d && belongs[i] != belongs[j]){
						min_d = data[i * size + j];
						node_a = i;
						node_b = j;
					}
				}
			}
			
			// are we done ?
			if (min_d > 0.5){
				int max_idx =0;
				for (int j = 0; j < size; j++){
					boolean visited = false;
					for(int i = 0; i < 2 * size - 1; i++){
						if (belongs[j] == i){
							// populate the correct / aggregated cluster
							indexes[j] = max_idx;
							visited = true;
						}
					}
					
					if (visited){
						max_idx++;
					}
				}
				
				// update the main map before going back
				for(int i = 0; i < size; i++){
					_boxClusterMap.put(dbb[i], indexes[i]);
				}
				return max_idx;
			}

			//4. Merge clusters and assign level
			if(node_a >= 0 && node_b >= 0){  // this should always BE true, otherwise we would have returned
				for (int k = 0; k < size; k++){
					if (belongs[k] == belongs[node_a] || belongs[k] == belongs[node_b])
						belongs[k] = m;
				}
				m++;
			}
		}
		
		// there seem to be only 1 cluster
		for(int i = 0; i < size; i++){
			_boxClusterMap.put(dbb[i], 0);
		}
		return 1;
	}
	
	
	/** Inputs:
	 * - Image
	 * - bad_boxes (Boxes far from the bounding box)
	 * - variance (pEx variance)
	 * Outputs
	 * - Negative fern features (nFerns)
	 * - Negative NN examples (nExample)
	 */
	private Pair<List<Pair<int[], Boolean>>, List<Mat>> generateNegativeData(final Mat frame){
		final List<Pair<int[], Boolean>> negFerns = new ArrayList<Pair<int[], Boolean>>();
		final List<Mat> negExamples = new ArrayList<Mat>();
		
		
		final List<BoundingBox> badBoxes = Arrays.asList(_grid.getBadBoxes());
		Collections.shuffle(badBoxes);
		Log.w(Util.TAG, "ST");
		// Get Fern Features of the boxes with big variance (calculated using integral images)
		for(BoundingBox badBox : badBoxes){
			if(Util.getVar(badBox, _iisumJava, _iisqsumJava, _iiCols) >= _var * 0.5f){
				final Mat patch = frame.submat(badBox);
				final int[] allFernsHashCodes = _classifierFern.getAllFernsHashCodes(patch, badBox.scaleIdx);
				negFerns.add(new Pair<int[], Boolean>(allFernsHashCodes, false));
			}
		}
		
		// select a hard coded number of negative examples
		Iterator<BoundingBox> bbIt = badBoxes.iterator();
		for(int i = 0; i < _params.num_bad_patches && bbIt.hasNext(); i++){
			final Mat pattern = new Mat();
			final Mat patch = frame.submat(bbIt.next());
			resizeZeroMeanStdev(patch, pattern, _params.patch_size);
			negExamples.add(pattern);
		}
		
		Log.i(Util.TAG, "Negative examples generated. Ferns count: " + negFerns.size() + ". negEx count: " + negExamples.size());
		
		return new Pair<List<Pair<int[],Boolean>>, List<Mat>>(negFerns, negExamples);
	}
	
	/**
	 * Generate Positive data 
	 * Inputs: 
	 * - good_boxes 
	 * - best_box 
	 * - bbhull
	 * Outputs: 
	 * - Positive fern features (pFerns) 
	 * - Positive NN examples (pExample)
	 */
	void generatePositiveData(final Mat frame, final int numWarps, final Grid aGrid) {
		resizeZeroMeanStdev(frame.submat(aGrid.getBestBox()), _pExample, _params.patch_size);
		//Get Fern features on warped patches
		final Mat img = new Mat();
		Imgproc.GaussianBlur(frame, img, new Size(9, 9), 1.5);
		final BoundingBox bbhull = aGrid.getBBhull();
		final Mat warped = img.submat(bbhull);
		// centre of the hull
		final Point pt = new Point(bbhull.x + (bbhull.width - 1) * 0.5f, bbhull.y + (bbhull.height - 1) * 0.5f);
		
		_pFerns.clear();
		_pPatterns.clear();
		
		for(int i = 0; i < numWarps; i++){
			if(i > 0){
				// this is important as it introduces the necessary noise / fuziness in the initial examples such that the Fern classifier recognises similar shapes not only Exact ones ! 
				// warped is a reference to a subset of the img data, so this will affect the img object
				_patchGenerator.generate(frame, pt, warped, bbhull.size(), _rng);
			}

			final BoundingBox[] goodBoxes = aGrid.getGoodBoxes();
			for(BoundingBox goodBox : goodBoxes){
				final Mat patch = img.submat(goodBox);
				final int[] allFernsHashCodes = _classifierFern.getAllFernsHashCodes(patch, goodBox.scaleIdx);
				_pFerns.add(new Pair<int[], Boolean>(allFernsHashCodes, true));
				
//				// this will be used for display only
//				final Mat tempPattern = new Mat();
//				Imgproc.resize(patch, tempPattern, new Size(_params.patch_size, _params.patch_size));
//				_pPatterns.add(tempPattern);
			}
		}
		
		Log.i(Util.TAG, "Positive examples generated( ferns: " + _pFerns.size() + " NN: 1/n )");
	}
	
	
	/**
	 * Output: resized zero-mean patch/pattern
	 * @param inImg INPUT, outPattern OUTPUT
	 * @return stdev
	 */
	private static double resizeZeroMeanStdev(final Mat inImg, Mat outPattern, int patternSize){
		if(inImg == null || outPattern == null){
			return -1;
		}
		
		Imgproc.resize(inImg, outPattern, new Size(patternSize, patternSize));
		final MatOfDouble mean = new MatOfDouble();
		final MatOfDouble stdev = new MatOfDouble();
		Core.meanStdDev(outPattern, mean, stdev);
		outPattern.convertTo(outPattern, CvType.CV_32F);
		Core.subtract(outPattern, new Scalar(mean.toArray()[0]), outPattern);
		
		return stdev.toArray()[0];
	}
	
	public List<Mat> getPPatterns(){
		return _pPatterns;
	}
	
	
	static final class DetectionStruct {
		public final BoundingBox detectedBB;
		public final int[] pattern;
		public final double averagePosterior;
		public final Mat patch;
		public NNConfStruct nnConf;
		
		DetectionStruct(BoundingBox detectedBB, int[] pattern, double averagePosterior, Mat patch) {
			this.detectedBB = detectedBB;
			this.pattern = pattern;
			this.averagePosterior = averagePosterior;
			this.patch = patch;
		}
	}
	
	private static final class TrackingStruct {
		public final float conf;
		public final BoundingBox predictedBB;
		public final Point[] lastPoints;
		public final Point[] currentPoints;
		
		TrackingStruct(float conf, BoundingBox predictedBB, Point[] trackedLastPoints, Point[] trackedCurrentPoints) {
			this.conf = conf;
			this.predictedBB = predictedBB;
			this.lastPoints = trackedLastPoints;
			this.currentPoints = trackedCurrentPoints;
		}
	}
	
	
	public static final class ProcessFrameStruct {
		public final Point[] lastPoints;
		public final Point[] currentPoints;
		public final BoundingBox currentBBox;
		
		ProcessFrameStruct(Point[] lastPoints, Point[] currentPoints, BoundingBox currentBBox) {
			this.lastPoints = lastPoints;
			this.currentPoints = currentPoints;
			this.currentBBox = currentBBox;
		}
	}
	
}
