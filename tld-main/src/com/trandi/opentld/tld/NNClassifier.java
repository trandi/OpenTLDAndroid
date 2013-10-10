package com.trandi.opentld.tld;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import android.util.Log;

import com.trandi.opentld.tld.Parameters.ParamsClassifiers;
import com.trandi.opentld.tld.Tld.DetectionStruct;
import com.trandi.opentld.tld.Util.IsinStruct;
import com.trandi.opentld.tld.Util.NNConfStruct;

/**
 * 
 * Nearest Neighbour classifier
 * 
 */
public class NNClassifier {
	ParamsClassifiers params;
	
	final List<Mat> pExamples = new ArrayList<Mat>();
	final List<Mat> nExamples = new ArrayList<Mat>();
	
	NNClassifier(Properties props) {
		params = new ParamsClassifiers(props);
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
			Log.e(Util.TAG, "NNClass.nnConf() - Null example received, stop here");
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
	 * Updates  NN threshold in case the negative threshold are above them.
	 * The Pos threshold has to be > to the negative one.
	 */
	void evaluateThreshold(final List<Mat> nExamplesTest){
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
	
	
	
	
	float getNNThreshold(){
		return params.pos_thr_nn;
	}
	
	float getNNThresholdValid(){
		return params.pos_thr_nn_valid;
	}
}
