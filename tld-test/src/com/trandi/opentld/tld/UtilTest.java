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
import java.util.Comparator;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;


public class UtilTest extends OpenCVTestCase {
	private static final int[] IISUM = new int[]{4, 65, 8, 23, 6, 98, 7, 2, 65, 44, 36, 74, 5, 12, 47, 86, 33, 4, 18, 51, 21, 36, 42, 78, 1};;
	private static final double[] IISQSUM = new double[]{16, 4225, 64, 529, 36, 9604, 49, 4, 4225, 1936, 1296, 5476, 25, 144, 2209, 7396, 1089, 16, 324, 2601, 441, 1296, 1764, 6084, 1};
	
	public void testKeepBestN(){
		final List<BoundingBox> boxes = new ArrayList<BoundingBox>();
		for(int i=1; i<=100; i++){
			final BoundingBox box = new BoundingBox(new Point(10 * i, i), new Point(20 * i, i + 10));
			box.overlap = i;
			boxes.add(box);
		}
		
		Util.keepBestN(boxes, 10, new Comparator<BoundingBox>(){
			@Override
			public int compare(BoundingBox box1, BoundingBox box2) {
				return Float.valueOf(box1.overlap).compareTo(box2.overlap);
			}
		});
		
		assertEquals(10, boxes.size());
		assertEquals(91f, boxes.get(0).overlap);
		assertEquals(100f, boxes.get(9).overlap);
	}
	
	
	public void testToByteArray(){
		final Mat greyMat = new Mat();
		Imgproc.cvtColor(getTestMat(), greyMat, Imgproc.COLOR_RGB2GRAY);
		// make sure we have some extreme values
		greyMat.put(0, 1, new byte[]{-128});
		greyMat.put(0, 2, new byte[]{-0});
		greyMat.put(0, 3, new byte[]{127});
		final byte[] array = Util.getByteArray(greyMat);
		
		assertEquals(2, array[0]);
		assertEquals(-128, array[1]);
		assertEquals(0, array[2]);
		assertEquals(127, array[3]);
		assertEquals(8, array[500]);
		assertEquals(9, array[1000]);
		assertEquals(9, array[1500]);
		assertEquals(17, array[2000]);
		assertEquals(18, array[3000]);
		assertEquals(3, array[4000]);
		assertEquals(8, array[5000]);
		assertEquals(9, array[6000]);
		assertEquals(3, array[7000]);
		assertEquals(5, array[8000]);
		assertEquals(15, array[9000]);
		assertEquals(9, array[12000]);
		assertEquals(4, array[15000]);
		
		final int cols = greyMat.cols();
		for(int row=0; row<greyMat.rows(); row++){
			for(int col=0; col<cols; col++){
				assertEquals(Util.getByte(row, col, greyMat), array[row * cols + col]);
			}
		}
	}
	
	
	/**
	 * This is OpenCV functionality, but just need to clarify for myself how it works.
	 * Does a Mat created with "submat()" still points to the same underlying data !?
	 */
	public void testSubmat(){
		final Mat greyMat = new Mat();
		Imgproc.cvtColor(getTestMat(), greyMat, Imgproc.COLOR_RGB2GRAY);
		final Mat submat = greyMat.submat(new Rect(0, 0, 10, 10));
		assertEquals(2, Util.getByte(0,  2, greyMat));
		assertEquals(2, Util.getByte(0,  2, submat));
		
		submat.put(0, 2, new byte[]{33});
		assertEquals(33, Util.getByte(0,  2, greyMat));
		assertEquals(33, Util.getByte(0,  2, submat));		
	}
	
	
	public void testGetVar(){
		final Mat img = new Mat(5, 5, CvType.CV_32SC1);
		img.put(0, 0, IISUM);
		
		final Mat img2 = new Mat(5, 5, CvType.CV_64F);
		img2.put(0, 0, IISQSUM);
		
		final BoundingBox test_box = new BoundingBox(1, 1, 3, 3, 1, 0);
		
		assertEquals("Wrong Var calculation", -417.55555, Util.getVar(test_box, img, img2), 0.00001);
	}
}
