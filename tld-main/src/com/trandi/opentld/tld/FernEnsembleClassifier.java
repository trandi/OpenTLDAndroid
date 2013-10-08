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

import java.util.List;
import java.util.Properties;

import org.opencv.core.Mat;
import org.opencv.core.Size;

import android.util.Log;

import com.trandi.opentld.tld.Parameters.ParamsClassifiers;
import com.trandi.opentld.tld.Util.Pair;
import com.trandi.opentld.tld.Util.RNG;


class FernEnsembleClassifier {
	ParamsClassifiers params;
	private Fern[] ferns;
	
//	final List<Mat> pExamples = new ArrayList<Mat>();
//	final List<Mat> nExamples = new ArrayList<Mat>();

	FernEnsembleClassifier(){
	}
	
	FernEnsembleClassifier(Properties props) {
		params = new ParamsClassifiers(props);
	}


	void init(Size[] scales, RNG rng){
		ferns = new Fern[params.numFerns];
		for(int i=0; i<ferns.length; i++){
			ferns[i] = new Fern(params.numFeaturesPerFern, scales, rng);
		}
	}
	
	
	/**
	 * Updates the POSITIVE Ferns
	 * The threshold for Positive results has to be > to the average of negative posteriors
	 */
	void evaluateThreshold(final List<Pair<int[], Boolean>> nFernsTest){
		for(Pair<int[], Boolean> fern : nFernsTest){
			// here we know/hope that fern.second is always FALSE, as they are all NEGATIVE examples
			final double averagePosterior = averagePosterior(fern.first);
			if(averagePosterior > params.pos_thr_fern){
				params.pos_thr_fern = averagePosterior;
			}
		}
	}
	
	
	void trainF(final List<Pair<int[], Boolean>> ferns, int resample){
		for(int i = 0; i < resample; i++){
			for(Pair<int[], Boolean> fern : ferns){
				// the THRESHOLDS are here to make sure we don't increase/decrease the probabilities beyond given limits, to give other hashCodes a chance
				if(fern.second){ // if it's a positive fern
					if(averagePosterior(fern.first) <= params.pos_thr_fern){
						updatePosteriors(fern.first, true);
					}
				}else if(averagePosterior(fern.first) >= params.neg_thr_fern){
					updatePosteriors(fern.first, false);
				}
			}
		}
	}
	
	private void updatePosteriors(final int[] fernsHashCodes, boolean positive){
		assert(params.numFerns == fernsHashCodes.length);
		
		for(int fern = 0; fern < fernsHashCodes.length; fern++){
			ferns[fern].addCountUpdatePosteriors(fernsHashCodes[fern], positive);
		}
	}
	

	/**
	 * @return conf
	 */
	double averagePosterior(final int[] fernsHashCodes){
		assert(params.numFerns == fernsHashCodes.length);
		
		double result = 0;
		for(int fern = 0; fern < fernsHashCodes.length; fern++){
			result += ferns[fern].posteriorProbabilities[fernsHashCodes[fern]];
		}
		return result / fernsHashCodes.length;
	}
	
	
	/**
	 * The numbers in this array can be up to 2^params.structSize as we shift left once of each feature
	 */
	int[] getAllFernsHashCodes(final Mat patch, int scaleIdx){
		final int[] result = new int[ferns.length];
		final byte[] imageData = Util.getByteArray(patch);
		final int cols = patch.cols();
		for(int fern = 0; fern < ferns.length; fern++){
			result[fern] = ferns[fern].calculateHashCode(scaleIdx, imageData, cols);
		}
		
		return result;
	}


	
	static class Fern {
		private final Feature[][] features; // per scaleIdx
		// per HASHCODE
		final double[] posteriorProbabilities;	// the probability that it's our image
		final long[] nCounter;  	// the number of NEGATIVE patches
		final long[] pCounter;		// the number of POSITIVE patches
		
		
		Fern(int featuresPerFern, Size[] scales, RNG rng) {
			// 1. Define random features
			features = new Feature[scales.length][featuresPerFern];
			for (int i=0; i<featuresPerFern; i++){
				final float x1f = rng.nextFloat();
				final float y1f = rng.nextFloat();
				final float x2f = rng.nextFloat();
				final float y2f = rng.nextFloat();
				for (int s=0; s<scales.length; s++){
					final int x1 = (int) (x1f * scales[s].width);
					final int y1 = (int) (y1f * scales[s].height);
					final int x2 = (int) (x2f * scales[s].width);
					final int y2 = (int) (y2f * scales[s].height);
					features[s][i] = new Feature(x1, y1, x2, y2);
				}
			}			
			
			// 2. Initialise Posteriors
			final int MAX_HASHCODE = (int)Math.pow(2d, featuresPerFern);
			posteriorProbabilities = new double[MAX_HASHCODE];
			pCounter = new long[MAX_HASHCODE];
			nCounter = new long[MAX_HASHCODE];
		}
		
		void addCountUpdatePosteriors(int fernHashCode, boolean positive) {
			if(positive){
				pCounter[fernHashCode] ++;
			}else{
				nCounter[fernHashCode] ++;
			}
			
			posteriorProbabilities[fernHashCode] = ((double)pCounter[fernHashCode]) / (pCounter[fernHashCode] + nCounter[fernHashCode]);
		}
		
		
		int calculateHashCode(int scaleIdx, byte[] imageData, int cols) {
			int fernHashCode = 0;
			for(Feature feature : features[scaleIdx]){
				// compare returns 0 / 1 and 
				fernHashCode = (fernHashCode << 1) + feature.compare(imageData, cols);
			}
			
			return fernHashCode;
		}
	}
	
	
	/**
	 * A Feature is a pixel Comparison, between 2 points.
	 */
	private static class Feature {
		private final int x1, y1, x2, y2;

		public Feature(int x1, int y1, int x2, int y2) {
			this.x1 = x1;
			this.y1 = y1;
			this.x2 = x2;
			this.y2 = y2;
		}

		/**
		 * Simply compares the brightness between the 2 points defining this Feature
		 * Assumes channels = 1 (hence only multiplying with cols).
		 */
		public int compare(final byte[] patch, final int cols) {
			final int pos1 = y1 * cols + x1;
			final int pos2 = y2 * cols + x2;
			if(pos1 >= patch.length || pos2 >= patch.length) {
				Log.w(Util.TAG, "Bad patch of size: " + patch.length + " cols: " + cols + " to compare Feature: " + this.toString());
				return 0;
			}
			
			final boolean boolRes = patch[pos1] > patch[pos2];
			return boolRes ? 1 : 0;
		}
		
		@Override
		public String toString(){
			return x1 + ", " + y1 + ", " + x2 + ", " + y2;
		}
	}
	
	
	int getNumFerns(){
		return params.numFerns;
	}
	
	double getFernPosThreshold(){
		return params.pos_thr_fern;
	}
	
	
	
	// TODO use to display the positive examples used by learning...
//	public Mat getPosExamples(){
//		if(pExamples == null || pExamples.size() == 0) return null;
//			
//		final int exRows = pExamples.get(0).rows();
//		final int exCols = pExamples.get(0).cols();
//		
//		// create a Matrix that can contain vertically all the positive examples
//		final Mat result = new Mat(pExamples.size() * exRows, exCols, CvType.CV_8U);
//		Imgproc.
//	}
}
