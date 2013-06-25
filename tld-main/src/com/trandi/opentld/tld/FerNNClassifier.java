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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.util.Log;

import com.trandi.opentld.tld.Parameters.ParamsClassifier;
import com.trandi.opentld.tld.Tld.DetectionStruct;
import com.trandi.opentld.tld.Util.IsinStruct;
import com.trandi.opentld.tld.Util.NNConfStruct;
import com.trandi.opentld.tld.Util.Pair;
import com.trandi.opentld.tld.Util.RNG;


/**
 * Fern Nearest Neighbours Classifier
 * 
 * @author trandi
 *
 */
class FerNNClassifier {
	ParamsClassifier params;
	private Feature[][] features;	//Ferns features (one array of totalFeatures for each scale)
	private float[][] posteriors;	//Ferns posteriors
	private int[][] nCounter;  		//negative counter
	private int[][] pCounter;		//positive counter
	
	final List<Mat> pExamples = new ArrayList<Mat>();
	final List<Mat> nExamples = new ArrayList<Mat>();

	FerNNClassifier(){
	}
	
	FerNNClassifier(Properties props) {
		params = new ParamsClassifier(props);
	}

	void prepare(Size[] scales, RNG rng){
		//Initialise test locations for features
		int totalFeatures = params.nstructs * params.structSize;
		features = new Feature[scales.length][totalFeatures];
		float x1f, x2f, y1f, y2f;
		int x1, x2, y1, y2;
		 
		for (int i=0; i<totalFeatures; i++){
			x1f = rng.nextFloat();
			y1f = rng.nextFloat();
			x2f = rng.nextFloat();
			y2f = rng.nextFloat();
			for (int s=0; s<scales.length; s++){
				x1 = (int) (x1f * scales[s].width);
				y1 = (int) (y1f * scales[s].height);
				x2 = (int) (x2f * scales[s].width);
				y2 = (int) (y2f * scales[s].height);
				features[s][i] = new Feature(x1, y1, x2, y2);
			}
		}
		

		//Initialise Posteriors
		final int fernsCount = (int)Math.pow(2d, params.structSize);
		posteriors = new float[params.nstructs][fernsCount];
		pCounter = new int[params.nstructs][fernsCount];
		nCounter = new int[params.nstructs][fernsCount];
	}
	
	
	/**
	 * Updates the POSITIVE Ferns and NN threshold in case the negative threshold are above them.
	 * The Pos threshold has to be > to the negative one.
	 */
	void evaluateThreshold(final List<Pair<int[], Boolean>> nFernsTest, final List<Mat> nExamplesTest){
		// Ferns
		for(Pair<int[], Boolean> fern : nFernsTest){
			final float fconf = measureForest(fern.first) / params.nstructs;
			if(fconf > params.pos_thr_fern){
				params.pos_thr_fern = fconf;
			}
		}
		
		// NN
		for(Mat ex : nExamplesTest){
			final NNConfStruct nnConf = nnConf(ex);
			if(nnConf.relativeSimilarity > params.pos_thr_nn){
				params.pos_thr_nn = nnConf.relativeSimilarity;
			}
		}
		if(params.pos_thr_nn > params.pos_thr_nn_valid){
			params.pos_thr_nn_valid = params.pos_thr_nn;
		}
	}
	
	
	void trainF(final List<Pair<int[], Boolean>> ferns, int resample){
		final float positiveThreshold = params.pos_thr_fern * params.nstructs;
		final float negativeThreshold = params.neg_thr_fern * params.nstructs;
		
//		for(int i = 0; i < resample; i++){
			for(Pair<int[], Boolean> fern : ferns){
				if(fern.second){ // if it's a positive fern
					if(measureForest(fern.first) <= positiveThreshold){
						update(fern.first, true);
					}
				}else if(measureForest(fern.first) >= negativeThreshold){
					update(fern.first, false);
				}
			}
//		}
	}
	
	/**
	 * OUTPUT (updates) : pExamples, nExamples
	 */
	void trainNN(final Mat pExampleIn, final List<Mat> nExamplesIn){
		NNConfStruct nnConf = nnConf(pExampleIn);
		if(nnConf.relativeSimilarity <= params.pos_thr_nn){
			if(nnConf.isin == null || nnConf.isin.idxPosSet < 0){
				pExamples.clear();
			}
			pExamples.add(pExampleIn);
		}
		
		for(Mat nEx : nExamplesIn){
			nnConf = nnConf(nEx);
			if(nnConf.relativeSimilarity > params.neg_thr_nn){
				nExamples.add(nEx);
			}
		}
		
		Log.i(Util.TAG, "Trained NN examples: " + pExamples.size() + " positive " + nExamples.size() + " negative");
	}
	
	
	/**
	 * INPUTs : pExamples, nExamples
	 * @param example NN patch
	 * @return Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
	 */
	NNConfStruct nnConf(final Mat example) {
		if(example == null){
			Log.e(Util.TAG, "Null example received, stop here");
			return new NNConfStruct(null, 0, 0);
		}
		if(pExamples.isEmpty()){
			// IF positive examples in the model are not defined THEN everything is negative
			return new NNConfStruct(null, 0, 0);
		}
		
		if(nExamples.isEmpty()){
			// IF negative examples in the model are not defined THEN everything is positive
			return new NNConfStruct(null, 1, 1);
		}
		
		final Mat ncc = new Mat(1, 1, CvType.CV_32F);
		float nccP=0, csmaxP=0, maxP=0;
		boolean anyP = false;
		int maxPidx = 0;
		final int validatedPart = (int) Math.ceil(pExamples.size() * params.valid);
		for(int i = 0; i < pExamples.size(); i++){
			Imgproc.matchTemplate(pExamples.get(i), example, ncc, Imgproc.TM_CCORR_NORMED);  // measure NCC to positive examples
			nccP = (Util.getFloat(0, 0, ncc) + 1) * 0.5f;
			if(nccP > params.ncc_thesame){
				anyP = true;
			}
			if(nccP > maxP){
				maxP = nccP;
				maxPidx = i;
				if(i < validatedPart){
					csmaxP = maxP;
				}
			}
		}
		
		float nccN=0, maxN = 0;
		boolean anyN = false;
		for(int i = 0; i < nExamples.size(); i++){
			Imgproc.matchTemplate(nExamples.get(i), example, ncc, Imgproc.TM_CCORR_NORMED);  //measure NCC to negative examples
			nccN = (Util.getFloat(0, 0, ncc) + 1) * 0.5f;
			if(nccN > params.ncc_thesame){
				anyN = true;
			}
			if(nccN > maxN){
				maxN = nccN;
			}
		}
		
		
		//Log.i(Util.TAG, "nccP=" + nccP + ", nccN=" + nccN + ", csmaxP=" + csmaxP + ", maxP="+ maxP + ", maxN=" + maxN);
		
		// put together the result
		final float dN = 1 - maxN;
		final float dPrelative = 1 - maxP;
		final float dPconservative = 1 - csmaxP;
		return new NNConfStruct(new IsinStruct(anyP, maxPidx, anyN), dN / (dN + dPrelative), dN / (dN + dPconservative));
	}
	
	
	/**
	 * @return conf
	 */
	float measureForest(final int[] fern){
		float result = 0;
		for(int i = 0; i < params.nstructs; i++){
			result += posteriors[i][fern[i]];
		}
		return result;
	}
	
	private void update(final int[] fern, boolean positive){
		for(int i = 0; i < params.nstructs; i++){
			final int idx = fern[i];
			if(positive){
				pCounter[i][idx] ++;
			}else{
				nCounter[i][idx] ++;
			}
			
			posteriors[i][idx] = (pCounter[i][idx] == 0 ? 0 : pCounter[i][idx] / (pCounter[i][idx] + nCounter[i][idx]));
		}
	}
	
	/**
	 * The numbers in this array can be up to 2^params.structSize as we shift left once of each feature
	 */
	int[] getFeatures(final Mat image, int scaleIdx){
		final int[] result = new int[params.nstructs];
		final byte[] imageData = Util.getByteArray(image);
		final int cols = image.cols();
		int leaf;
		for(int tree = 0; tree < params.nstructs; tree++){
			leaf = 0;
			for(int feature = 0; feature < params.structSize; feature++){
				// compare returns 0 / 1 and 
				leaf = (leaf << 1) + features[scaleIdx][tree * params.nstructs + feature].compare(imageData, cols);
			}
			result[tree] = leaf;
		}
		
		return result;
	}

	
	/**
	 * @return Those boxes that pass the Nearest neighbour test/threshold, and their Conservative confidence
	 */
	Map<BoundingBox, Float> getNnValidBoxes(final List<DetectionStruct> list) {
		final Map<BoundingBox, Float> result = new HashMap<BoundingBox, Float>();
		for(DetectionStruct detStruct : list){
			if(detStruct.nnConf.relativeSimilarity > getNNThreshold()){
				result.put(detStruct.detectedBB, detStruct.nnConf.conservativeSimilarity);
			}
		}
		
		return result;
	}
	
	
	private static class Feature {
		private final int x1, y1, x2, y2;

		public Feature(int x1, int y1, int x2, int y2) {
			this.x1 = x1;
			this.y1 = y1;
			this.x2 = x2;
			this.y2 = y2;
		}

		/**
		 * Assumes channels = 1 (hence only multiplying with cols)
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
	
	
	int getNumStructs(){
		return params.nstructs;
	}
	
	float getFernThreshold(){
		return params.pos_thr_fern;
	}
	
	float getNNThreshold(){
		return params.pos_thr_nn;
	}
	
	float getNNThresholdValid(){
		return params.pos_thr_nn_valid;
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
