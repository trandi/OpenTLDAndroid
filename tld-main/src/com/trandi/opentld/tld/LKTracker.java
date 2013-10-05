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
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import android.util.Log;

import com.trandi.opentld.tld.Util.Pair;

class LKTracker {
	private static final int MAX_COUNT = 20;
	private static final double EPSILON = 0.03;
	private static final Size WINDOW_SIZE = new Size(4, 4);
	private static final int MAX_LEVEL = 5;
	private static final float LAMBDA = 0f; // minEigenThreshold
	private static final Size CROSS_CORR_PATCH_SIZE = new Size(10, 10);
	
	private final TermCriteria termCriteria;
	float errFBMed;
	

	
	LKTracker(){
		termCriteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, MAX_COUNT, EPSILON);
	}
	
	
	/**
	 * @return Pair of new, FILTERED, last and current POINTS, or null if it hasn't managed to track anything.
	 */
	Pair<Point[], Point[]> track(final Mat lastImg, final Mat currentImg, Point[] lastPoints){
		final int size = lastPoints.length;
		final MatOfPoint2f currentPointsMat = new MatOfPoint2f();
		final MatOfPoint2f pointsFBMat = new MatOfPoint2f();
		final MatOfByte statusMat = new MatOfByte();
		final MatOfFloat errSimilarityMat = new MatOfFloat();
		final MatOfByte statusFBMat = new MatOfByte();
		final MatOfFloat errSimilarityFBMat = new MatOfFloat();
		
		//Forward-Backward tracking
		Video.calcOpticalFlowPyrLK(lastImg, currentImg, new MatOfPoint2f(lastPoints), currentPointsMat, 
				statusMat, errSimilarityMat, WINDOW_SIZE, MAX_LEVEL, termCriteria, 0, LAMBDA);
		Video.calcOpticalFlowPyrLK(currentImg, lastImg, currentPointsMat, pointsFBMat, 
				statusFBMat, errSimilarityFBMat, WINDOW_SIZE, MAX_LEVEL, termCriteria, 0, LAMBDA);
		
		final byte[] status = statusMat.toArray();
		float[] errSimilarity = new float[lastPoints.length]; 
		//final byte[] statusFB = statusFBMat.toArray();
		final float[] errSimilarityFB = errSimilarityFBMat.toArray();	
		
		// compute the real FB error (relative to LAST points not the current ones...
		final Point[] pointsFB = pointsFBMat.toArray();
		for(int i = 0; i < size; i++){
			errSimilarityFB[i] = Util.norm(pointsFB[i], lastPoints[i]);
		}
		
		final Point[] currPoints = currentPointsMat.toArray();
		// compute real similarity error
		errSimilarity = normCrossCorrelation(lastImg, currentImg, lastPoints, currPoints, status);
		
		
		//TODO  errSimilarityFB has problem != from C++
		// filter out points with fwd-back error > the median AND points with similarity error > median
		return filterPts(lastPoints, currPoints, errSimilarity, errSimilarityFB, status);
	}
	
	
	/**
	 * @return real similarities errors
	 */
	private float[] normCrossCorrelation(final Mat lastImg, final Mat currentImg, final Point[] lastPoints, final Point[] currentPoints, final byte[] status){
		final float[] similarity = new float[lastPoints.length];
		
		final Mat lastPatch = new Mat(CROSS_CORR_PATCH_SIZE, CvType.CV_8U);
		final Mat currentPatch = new Mat(CROSS_CORR_PATCH_SIZE, CvType.CV_8U);
		final Mat res = new Mat(new Size(1, 1), CvType.CV_32F);
		
		for(int i = 0; i < lastPoints.length; i++){
			if(status[i] == 1){
				Imgproc.getRectSubPix(lastImg, CROSS_CORR_PATCH_SIZE, lastPoints[i], lastPatch);
				Imgproc.getRectSubPix(currentImg, CROSS_CORR_PATCH_SIZE, currentPoints[i], currentPatch);
				Imgproc.matchTemplate(lastPatch, currentPatch, res, Imgproc.TM_CCOEFF_NORMED);
				
				similarity[i] = Util.getFloat(0, 0, res);
			}else{
				similarity[i] = 0f;
			}
		}
		
		return similarity;
	}
	
	
	/**
	 * @return Pair of new, FILTERED, last and current POINTS. Null if none were valid (with similarity > median and FB error <= median)
	 */
	private Pair<Point[], Point[]> filterPts(final Point[] lastPoints, final Point[] currentPoints, final float[] similarity, final float[] errFB, final byte[] status){
		final List<Point> filteredLastPoints = new ArrayList<Point>();
		final List<Point> filteredCurrentPoints = new ArrayList<Point>();
		final List<Float> filteredErrFB = new ArrayList<Float>();
		
		final float similarityMed = Util.median(similarity);
		Log.i(Util.TAG, "Filter points MED SIMILARITY: " + similarityMed);
		
		for(int i = 0; i < currentPoints.length; i++){
			if(status[i] == 1 && similarity[i] > similarityMed){
				filteredLastPoints.add(lastPoints[i]);
				filteredCurrentPoints.add(currentPoints[i]);
				filteredErrFB.add(errFB[i]);
			}
		}
		
		final List<Point> filteredLastPoints2 = new ArrayList<Point>();
		final List<Point> filteredCurrentPoints2 = new ArrayList<Point>();
		if(filteredErrFB.size() > 0){
			errFBMed = Util.median(filteredErrFB);
			
			for(int i = 0; i < filteredErrFB.size(); i++){
				// status has already been checked
				if(filteredErrFB.get(i) <= errFBMed){
					filteredLastPoints2.add(filteredLastPoints.get(i));
					filteredCurrentPoints2.add(filteredCurrentPoints.get(i));
				}
			}
			
			Log.i(Util.TAG, "Filter points MED ErrFB: " + errFBMed + " K count=" + filteredLastPoints2.size());
		}
		
		final int size = filteredLastPoints2.size();
		return size > 0 ? new Pair<Point[], Point[]>(filteredLastPoints2.toArray(new Point[size]), filteredCurrentPoints2.toArray(new Point[size])) : null;
	}
	
	float getMedianErrFB(){
		return errFBMed;
	}
}
