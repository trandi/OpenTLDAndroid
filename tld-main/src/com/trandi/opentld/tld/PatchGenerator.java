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

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.trandi.opentld.tld.Util.RNG;

class PatchGenerator {
	final double backgroundMin; 
	final double backgroundMax; 
	final double noiseRange; 
	final boolean randomBlur;
	final double lambdaMin; 
	final double lambdaMax; 
	final double thetaMin; 
	final double thetaMax; 
	final double phiMin; 
	final double phiMax;	
	

	PatchGenerator(double backgroundMin, double backgroundMax, double noiseRange, boolean randomBlur,
            double lambdaMin, double lambdaMax, double thetaMin, double thetaMax, double phiMin, double phiMax )
	{
		this.backgroundMin = backgroundMin; 
		this.backgroundMax = backgroundMax; 
		this.noiseRange = noiseRange; 
		this.randomBlur = randomBlur;
		this.lambdaMin = lambdaMin; 
		this.lambdaMax = lambdaMax; 
		this.thetaMin = thetaMin; 
		this.thetaMax = thetaMax; 
		this.phiMin = phiMin; 
		this.phiMax = phiMax;
	}
	
	void generate(final Mat image, Point pt, Mat patch, Size patchSize, final RNG rng) {
		final Mat T = new MatOfDouble();
		
		// TODO why is inverse not specified in the original C++ code
		generateRandomTransform(pt, new Point((patchSize.width - 1) * 0.5, (patchSize.height - 1) * 0.5), T, false);
		
		generate(image, T, patch, patchSize, rng);
	}
	
	
	/**
	 * 
	 * @param image
	 * @param T
	 * @param patch OUTPUT
	 * @param patchSize
	 */
	void generate(final Mat image, final Mat T, Mat patch, Size patchSize, final RNG rng){
	    patch.create( patchSize, image.type() );
	    if( backgroundMin != backgroundMax ) {
	    	Core.randu(patch, backgroundMin, backgroundMax);
	    	// TODO if that null scalar OK or should it be new Scalar(0) ?
	    	Imgproc.warpAffine(image, patch, T, patchSize, Imgproc.INTER_LINEAR, Imgproc.BORDER_TRANSPARENT, null);
	    } else {
	    	Imgproc.warpAffine(image, patch, T, patchSize, Imgproc.INTER_LINEAR, Imgproc.BORDER_CONSTANT, new Scalar(backgroundMin));
	    }

	    int ksize = randomBlur ? rng.nextInt() % 9 - 5 : 0;
	    if( ksize > 0 ) {
	        ksize = ksize * 2 + 1;
	        Imgproc.GaussianBlur(patch, patch, new Size(ksize, ksize), 0, 0);
	    }

	    if( noiseRange > 0 ) {
	        final Mat noise = new Mat(patchSize, image.type());
	        int delta = (image.depth() == CvType.CV_8U ? 128 : (image.depth() == CvType.CV_16U ? 32768 : 0));
	        Core.randn(noise, delta, noiseRange);
	        
	        // TODO this was different !!
	        Core.addWeighted(patch, 1, noise, 1, -delta, patch);
	        
//	        if( backgroundMin != backgroundMax )
//	            addWeighted(patch, 1, noise, 1, -delta, patch);
//	        else
//	        {
//	            for( int i = 0; i < patchSize.height; i++ )
//	            {
//	                uchar* prow = patch.ptr<uchar>(i);
//	                const uchar* nrow =  noise.ptr<uchar>(i);
//	                for( int j = 0; j < patchSize.width; j++ )
//	                    if( prow[j] != backgroundMin )
//	                        prow[j] = saturate_cast<uchar>(prow[j] + nrow[j] - delta);
//	            }
//	        }
	    }		
	}
	
	
	
	/**
	 * 
	 * @param srcCenter
	 * @param dstCenter
	 * @param transform OUTPUT
	 * @param inverse
	 */
	private void generateRandomTransform(Point srcCenter, Point dstCenter, Mat transform, boolean inverse) {
		MatOfDouble tempRand = new MatOfDouble(0d, 0d);
		Core.randu(tempRand, lambdaMin, lambdaMax);
		final double[] rands = tempRand.toArray();
		final double lambda1 = rands[0];
		final double lambda2 = rands[1];
		Core.randu(tempRand, thetaMin, thetaMax);
		final double theta = tempRand.toArray()[0];
		Core.randu(tempRand, phiMin, phiMax);
		final double phi = tempRand.toArray()[0];
		
		
		// Calculate random parameterized affine transformation A,
	    // A = T(patch center) * R(theta) * R(phi)' * S(lambda1, lambda2) * R(phi) * T(-pt)
	    final double st = Math.sin(theta);
	    final double ct = Math.cos(theta);
	    final double sp = Math.sin(phi);
	    final double cp = Math.cos(phi);
	    final double c2p = cp*cp;
	    final double s2p = sp*sp;

	    final double A = lambda1*c2p + lambda2*s2p;
	    final double B = (lambda2 - lambda1)*sp*cp;
	    final double C = lambda1*s2p + lambda2*c2p;

	    final double Ax_plus_By = A*srcCenter.x + B*srcCenter.y;
	    final double Bx_plus_Cy = B*srcCenter.x + C*srcCenter.y;

	    transform.create(2, 3, CvType.CV_64F);
	    transform.put(0, 0, A*ct - B*st);
	    transform.put(0, 1, B*ct - C*st);
	    transform.put(0, 2, -ct*Ax_plus_By + st*Bx_plus_Cy + dstCenter.x);
	    transform.put(1, 0, A*st + B*ct);
	    transform.put(1, 1, B*st + C*ct);
	    transform.put(1, 2, -st*Ax_plus_By - ct*Bx_plus_Cy + dstCenter.y);

	    if( inverse ){
	        Imgproc.invertAffineTransform(transform, transform);
	    }
	}
}
