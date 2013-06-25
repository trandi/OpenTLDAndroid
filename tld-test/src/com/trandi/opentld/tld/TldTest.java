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

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import com.trandi.opentld.tld.Parameters.ParamsTld;


public class TldTest extends OpenCVTestCase {
	private static final float[] P_EXAMPLE = new float[]
			{-17.6667f, -3.66667f, 13.3333f, 
			 12.3333f, -22.6667f, 10.3333f, 
			 2.33333f, 19.3333f, -13.6667f};
	
	private static final float[] PATTERN = new float[]
			{10.2222f, -22.7778f, -3.77778f, 
			 20.2222f, -23.7778f, 6.22222f, 
			 11.2222f, 0.222222f, 2.22222f};
	private static final int[] EXPECTED_FERN = new int[]{4384, 84, 4290, 2065, 1237, 5719, 7926, 6400, 265, 1037};
	
	
	
	
	public void testGetPattern(){
		final int patch_size = 3;
		
		final Mat img = getSimpleMat();
		// manual init
		final Tld tld = new Tld();
		tld._params = new DummyParamsTld(1, patch_size);
		Mat pattern = new Mat(patch_size, patch_size, CvType.CV_64F);
		final double stdev = tld.getPattern(img, pattern);
		
		final float[] patternF = Util.getFloatArray(pattern);
		for(int i=0; i<PATTERN.length; i++){
			assertEquals("Wrong element " + i + " in the pattern", PATTERN[i], patternF[i], 0.0001);
		}
		
		assertEquals("Wrong STDEV for the pattern", 14.0695f, stdev, 0.0001);
	}
	
	public void testGenerateNegativeData(){
		//TODO
	}
	
	
	public void testGeneratePositiveData(){
		final int min_win = 1;
		final int patch_size = 3;
		final int num_warps = 10;
		final int nstructs = 10;
		final int structSize = 13;
		final BoundingBox best_box = new BoundingBox(1, 1, 4, 4, 1, 0);
	
		// test matrix/image
		final Mat img = getSimpleMat();
	
		// manual init
		final Tld tld = new Tld();
		tld._params = new DummyParamsTld(min_win, patch_size);

		tld._pExample.create(patch_size, patch_size, CvType.CV_64F);
		
		final Grid grid = new Grid();
		grid.bestBox = best_box;
		grid.bbHull = best_box; // only 1 box actually
		grid.goodBoxes.add(best_box);
		grid.grid.add(best_box);
		
		final Size[] scales = new Size[]{new Size(2, 2)};
		tld._classifier = new FerNNClassifier();
		tld._classifier.params = new DummyParamsClassifier(nstructs, structSize);
		tld._classifier.prepare(scales, new DummyRNG());
		
		//tld.patchGenerator = new DummyPatchGenerator();

		// run
		tld.generatePositiveData(img, num_warps, grid);
		
		// Check Positive Examples
		final float[] pExample = Util.getFloatArray(tld._pExample);
		for(int i=0; i<P_EXAMPLE.length; i++){
			assertEquals("Wrong pExample item " + i, P_EXAMPLE[i], pExample[i], 0.0001);
		}
		
		// Check Positive Ferns
		final int[] fern = tld._pFerns.get(0).first;
		for(int i=0; i<fern.length; i++){
			assertEquals("Different element in fern", EXPECTED_FERN[i], fern[i]);
		}
	}
	
	
	
	private static class DummyParamsTld extends ParamsTld{
		DummyParamsTld(int min_win, int patch_size) {
			this.min_win = min_win;
			this.patch_size = patch_size;
		}
	}
}
