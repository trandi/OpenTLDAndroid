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

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;

import com.trandi.opentld.tld.Util.NNConfStruct;

public class FerNNClassifierTest extends OpenCVTestCase{
	static int[] EXPECTED_FERN = new int[]{6278, 6386, 2241, 1139, 3291, 3927, 7902, 6144, 256, 14};
	
	public void testGetFeatures(){
		final Size[] scales = new Size[]{new Size(2, 2)};
		FerNNClassifier classifier = new FerNNClassifier();
		classifier.params = new DummyParamsClassifier(10, 13);
		classifier.prepare(scales, new DummyRNG());
		
		final int[] fern = classifier.getFeatures(getSimpleMat(), 0);
		for(int i=0; i<fern.length; i++){
			assertEquals("Different element in fern", EXPECTED_FERN[i], fern[i]);
		}
	}
	
	
	
	
	private static float[] IMG = new float[]{-1.88f, -1.88f, -1.88f, -1.88f, -0.88f,
	  -1.88f, -1.88f, -1.88f, -1.88f, -1.88f,
	  4.1199999f, -1.88f, -1.88f, -1.88f, -1.88f,
	  4.1199999f, 0.12f, -1.88f, -0.88f, 0.12f,
	  -0.88f, 0.12f, -1.88f, -1.88f, 24.120001f};
	private static float[] PEX1 = new float[]{-0.75999999f, -0.75999999f, 3.24f, -0.75999999f, -0.75999999f,
	  -0.75999999f, -0.75999999f, 1.24f, -0.75999999f, -0.75999999f,
	  1.24f, 0.23999999f, 0.23999999f, 0.23999999f, 0.23999999f,
	  0.23999999f, 1.24f, -0.75999999f, 0.23999999f, -0.75999999f,
	  0.23999999f, 0.23999999f, 0.23999999f, -0.75999999f, -0.75999999f};
	private static float[] PEX2 = new float[]{-1f, 0f, 2f, -1f, -1f,
	  -1f, -1f, -1f, -1f, -1f,
	  1f, 1f, -1f, -1f, -1f,
	  -1f, 3f, -1f, -1f, -1f,
	  -1f, 2f, -1f, -1f, 9f};
	private static float[] PEX3 = new float[]{-2.3199999f, -2.3199999f, -2.3199999f, -2.3199999f, -2.3199999f,
	  0.68000001f, -0.31999999f, -1.3200001f, -2.3199999f, -2.3199999f,
	  4.6799998f, 1.6799999f, -1.3200001f, -1.3200001f, -1.3200001f,
	  2.6800001f, -1.3200001f, -2.3199999f, -2.3199999f, 0.68000001f,
	  -0.31999999f, -0.31999999f, -2.3199999f, -2.3199999f, 22.68f};

	private static float[] NEX1 = new float[]{-10.24f, -11.24f, 8.7600002f, 7.7600002f, 10.76f,
	  -10.24f, -11.24f, 10.76f, 7.7600002f, 7.7600002f,
	  -11.24f, -12.24f, 5.7600002f, 6.7600002f, 8.7600002f,
	  -13.24f, -10.24f, 5.7600002f, 5.7600002f, 5.7600002f,
	  -11.24f, -10.24f, 6.7600002f, 6.7600002f, 5.7600002f};
	private static float[] NEX2 = new float[]{2.96f, -6.04f, -6.04f, -4.04f, -7.04f,
	  1.96f, 0.95999998f, 1.96f, 2.96f, 4.96f,
	  2.96f, -0.039999999f, 1.96f, 4.96f, 3.96f,
	  2.96f, -1.04f, -1.04f, 4.96f, -3.04f,
	  0.95999998f, -4.04f, -3.04f, -0.039999999f, -3.04f};
	
	public void testNNConf(){
		Mat img = new MatOfFloat(IMG);
		Mat pEx1 = new MatOfFloat(PEX1);
		Mat pEx2 = new MatOfFloat(PEX2);
		Mat pEx3 = new MatOfFloat(PEX3);
		Mat nEx1 = new MatOfFloat(NEX1);
		Mat nEx2 = new MatOfFloat(NEX2);


		FerNNClassifier classifier = new FerNNClassifier();
		classifier.params = new DummyParamsClassifier(0.5f, 0.95f, -1f, -1f);
		classifier.pExamples.add(pEx1);
		classifier.pExamples.add(pEx2);
		classifier.pExamples.add(pEx3);
		classifier.nExamples.add(nEx1);
		classifier.nExamples.add(nEx2);
		NNConfStruct confStruct = classifier.nnConf(img);


		assertEquals(0.97415, confStruct.relativeSimilarity, 0.00001);
		assertEquals(0.86937, confStruct.conservativeSimilarity, 0.00001);
		assertEquals(true, confStruct.isin.inPosSet);
		assertEquals(2, confStruct.isin.idxPosSet);
		assertEquals(false, confStruct.isin.inNegSet);
	}
	
	public void testTrainNN(){
		Mat pEx1 = new MatOfFloat(PEX1);
		Mat pEx2 = new MatOfFloat(PEX2);
		Mat pEx3 = new MatOfFloat(PEX3);
		Mat nEx1 = new MatOfFloat(NEX1);
		Mat nEx2 = new MatOfFloat(NEX2);


		FerNNClassifier classifier = new FerNNClassifier();
		classifier.params = new DummyParamsClassifier(0.5f, 0.95f, 0.65f, 0.5f);
		List<Mat> nExamples = new ArrayList<Mat>();
		nExamples.add(pEx2);
		nExamples.add(pEx3);
		nExamples.add(nEx1);
		nExamples.add(nEx2);
		classifier.trainNN(pEx1, nExamples);
		assertEquals(1, classifier.pExamples.size());
		assertEquals(3, classifier.nExamples.size());
	}
}
