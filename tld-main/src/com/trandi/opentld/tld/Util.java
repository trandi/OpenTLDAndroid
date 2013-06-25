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

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;

public class Util {
	public final static String TAG = "OpenTLD";
	
	private final static byte[] _byteBuff1 = new byte[1];
	private final static int[] _intBuff1 = new int[1];
	private final static float[] _floatBuff1 = new float[1];
	private final static double[] _doubleBuff1 = new double[1];
	// this is for a whole matrix !
	private static byte[] _byteBuff = new byte[1];
	private static int[] _intBuff = new int[1];
	private static float[] _floatBuff = new float[1];
	private static double[] _doubleBuff = new double[1];
	
	
	static int unsignedChar(int val){
		return Math.min(Math.max(val, 0), 255);
	}
	
	static int unsignedChar(double val){
		return unsignedChar((int)val);
	}
	
	
	/**
	 * This is actually quite slow, as it makes a couple of native calls for each of the Util.getXXX methods...
	 */
	@Deprecated
	static double getVar(final BoundingBox box, final Mat sum, final Mat sqsum){
		final int brs = Util.getInt(box.y + box.height, box.x + box.width, sum);
		final int bls = Util.getInt(box.y + box.height, box.x, sum);
		final int trs = Util.getInt(box.y, box.x + box.width, sum);
		final int tls = Util.getInt(box.y, box.x, sum);
		final double brsq = Util.getDouble(box.y + box.height, box.x + box.width, sqsum);
		final double blsq = Util.getDouble(box.y + box.height, box.x, sqsum);
		final double trsq = Util.getDouble(box.y, box.x + box.width, sqsum);
		final double tlsq = Util.getDouble(box.y, box.x, sqsum);
		
		final double boxArea = box.area();
		final double mean = (brs + tls - trs - bls) / boxArea;
		final double sqmean = (brsq + tlsq - trsq - blsq) / boxArea;
		
		return sqmean - mean * mean;
	}
	
	/**
	 * Preferred for performance !
	 * Here we get both the SUM and SQuaredSUM Matrices already in Java, rather than call native methods for each 
	 * element that we need.
	 * 
	 * For a 320x240 frame the improvement is close to 5X !!!
	 */
	static double getVar(final BoundingBox box, final int[] sum, final double[] sqsum, final int colCount) {
		final int brs = sum[(box.y + box.height) * colCount + box.x + box.width];
		final int bls = sum[(box.y + box.height) * colCount + box.x];
		final int trs = sum[box.y * colCount + box.x + box.width];
		final int tls = sum[box.y * colCount + box.x];
		final double brsq = sqsum[(box.y + box.height) * colCount + box.x + box.width];
		final double blsq = sqsum[(box.y + box.height) * colCount + box.x];
		final double trsq = sqsum[box.y * colCount + box.x + box.width];
		final double tlsq = sqsum[box.y * colCount + box.x];
		
		final double boxArea = box.area();
		final double mean = (brs + tls - trs - bls) / boxArea;
		final double sqmean = (brsq + tlsq - trsq - blsq) / boxArea;
		
		return sqmean - mean * mean;		
	}
	
	static float median(float[] vals){
		final float[] newVals = Arrays.copyOf(vals, vals.length);
		Arrays.sort(newVals);
		return newVals[newVals.length / 2];
	}
	
	static float median(List<Float> vals){
		final Float[] newVals = vals.toArray(new Float[vals.size()]);
		Arrays.sort(newVals);
		return newVals[(int) Math.floor(newVals.length / 2d)];
	}
	
	static float norm(final Point p1, final Point p2){
		final double dX = p1.x - p2.x;
		final double dY = p1.y - p2.y;
		return (float)Math.sqrt(dX * dX + dY * dY);
	}
	
	/**
	 * no std::nth_element in Java so we'll sort the list. Less performant but we don't really care for the small lists we have 
	 */
	static <T> void keepBestN(List<T> list, final int n, final Comparator<T> comparator){
		final int size = list.size();
		if(size <= n) {
			// nothing to do, sorting is not a requirement
			return;
		}
		
		// sorts in ASCENDING ORDER
		Collections.sort(list, comparator);
		// we want the best / highest n so remote at the queue
		while(list.size() > n){
			list.remove(0);
		}
	}
	
	
	static byte getByte(final int row, final int col, final Mat mat){
		if(CvType.CV_8UC1 != mat.type()) throw new IllegalArgumentException("Expected type is CV_8UC1, we found: " + CvType.typeToString(mat.type()));
		
		mat.get(row, col, _byteBuff1);
		return _byteBuff1[0];
	}
	
	/**
	 * The corresponding Java primitive array type depends on the Mat type:
	 * CV_8U and CV_8S -> byte[]
	 * CV_16U and CV_16S -> short[]
	 * CV_32S -> int[]
	 * CV_32F -> float[]
	 * CV_64F-> double[]
	 */
	static byte[] getByteArray(final Mat mat){
		if(CvType.CV_8UC1 != mat.type()) throw new IllegalArgumentException("Expected type is CV_8UC1, we found: " + CvType.typeToString(mat.type()));
		
		final int size = (int) (mat.total() * mat.channels());
		if(_byteBuff.length != size){
			_byteBuff = new byte[size];
		}
		mat.get(0, 0, _byteBuff); // 0 for row and col means the WHOLE Matrix
		return _byteBuff;
	}
	
	static int[] getIntArray(final Mat mat){
		if(CvType.CV_32SC1 != mat.type()) throw new IllegalArgumentException("Expected type is CV_32SC1, we found: " + CvType.typeToString(mat.type()));
		
		final int size = (int) (mat.total() * mat.channels());
		if(_intBuff.length != size){
			_intBuff = new int[size];
		}
		mat.get(0, 0, _intBuff); // 0 for row and col means the WHOLE Matrix
		return _intBuff;
	}
	
	static float[] getFloatArray(final Mat mat){
		if(CvType.CV_32FC1 != mat.type()) throw new IllegalArgumentException("Expected type is CV_32FC1, we found: " + CvType.typeToString(mat.type()));
		
		final int size = (int) (mat.total() * mat.channels());
		if(_floatBuff.length != size){
			_floatBuff = new float[size];
		}
		mat.get(0, 0, _floatBuff); // 0 for row and col means the WHOLE Matrix
		return _floatBuff;
	}
	
	static double[] getDoubleArray(final Mat mat){
		if(CvType.CV_64F != mat.type()) throw new IllegalArgumentException("Expected type is CV_64F, we found: " + CvType.typeToString(mat.type()));
		
		final int size = (int) (mat.total() * mat.channels());
		if(_doubleBuff.length != size){
			_doubleBuff = new double[size];
		}
		mat.get(0, 0, _doubleBuff); // 0 for row and col means the WHOLE Matrix
		return _doubleBuff;
	}
	
	
	static int getInt(final int row, final int col, final Mat mat){
		if(CvType.CV_32SC1 != mat.type()) throw new IllegalArgumentException("Expected type is CV_32SC1, we found: " + CvType.typeToString(mat.type()));
		
		mat.get(row, col, _intBuff1);
		return _intBuff1[0];
	}
	
	static float getFloat(final int row, final int col, final Mat mat){
		if(CvType.CV_32F != mat.type()) throw new IllegalArgumentException("Expected type is CV_32F, we found: " + CvType.typeToString(mat.type()));
		
		mat.get(row, col, _floatBuff1);
		return _floatBuff1[0];
	}
	
	static double getDouble(final int row, final int col, final Mat mat){
		if(CvType.CV_64F != mat.type()) throw new IllegalArgumentException("Expected type is CV_64F, we found: " + CvType.typeToString(mat.type()));
		
		mat.get(row, col, _doubleBuff1);
		return _doubleBuff1[0];
	}
	
	
	static final class NNConfStruct {
		final IsinStruct isin;
		final float relativeSimilarity;
		final float conservativeSimilarity;
		
		
		NNConfStruct(IsinStruct isin, float relativeSimilarity, float conservativeSimilarity) {
			this.isin = isin;
			this.relativeSimilarity = relativeSimilarity;
			this.conservativeSimilarity = conservativeSimilarity;
		}
	}
	
	static final class IsinStruct {
		final boolean inPosSet;
		final int idxPosSet;
		final boolean inNegSet;
		
		
		IsinStruct(boolean inPosSet, int idxPosSet, boolean inNegSet) {
			this.inPosSet = inPosSet;
			this.idxPosSet = idxPosSet;
			this.inNegSet = inNegSet;
		}
	}
	
	
	static final class Pair<U, V>{
		final U first;
		final V second;
		
		Pair(U first, V second){
			this.first = first;
			this.second = second;
		}
		
		@Override
		public String toString(){
			return "{" + first + ", " + second + "}";
		}
	}
	
	
	static interface RNG {
		float nextFloat();
		int nextInt();
	}
	
	static class DefaultRNG implements RNG{
		private final Random rnd = new Random();
		
		@Override
		public float nextFloat() {
			return rnd.nextFloat();
		}

		@Override
		public int nextInt() {
			return rnd.nextInt();
		}
		
	}
}
