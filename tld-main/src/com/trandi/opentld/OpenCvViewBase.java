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

package com.trandi.opentld;

import java.util.List;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.BitmapFactory.Options;
import android.graphics.Canvas;
import android.os.Debug;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import com.trandi.opentld.tld.Util;

public abstract class OpenCvViewBase extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    final protected SurfaceHolder       _holder;
    private VideoCapture        _camera;
    final private FpsMeter      _fpsMeter = new FpsMeter();
    private int _frameWidth = -1;
    private int _frameHeight = -1;
    
    protected int _canvasImgYOffset;
    protected int _canvasImgXOffset;
    
    
    
    /**
     * Implement this in the subclass and play with the capture.
     */
    protected abstract Bitmap processFrame(Mat frame);
    
    
    public OpenCvViewBase(Context context) {
        super(context);
        _holder = getHolder();
        _holder.addCallback(this);
        Log.i(Util.TAG, "Instantiated new " + this.getClass());
    }

    boolean openCamera(){
    	Log.i(Util.TAG, "openCamera");
    	synchronized (this){
    		//releaseCamera();
    		_camera = new VideoCapture(Highgui.CV_CAP_ANDROID);
    		if(!_camera.isOpened()){
    			releaseCamera();
    			Log.e(Util.TAG, "Failed to open native camera");
    			return false;
    		}
    		

    	}
    	return true;
    }
    
    void releaseCamera(){
    	Log.i(Util.TAG, "releaseCamera");
    	synchronized (this){
    		if(_camera != null){
    			_camera.release();
    			_camera = null;
    		}
    	}
    }
    
    private void setupCamera(int width, int height){
    	_camera.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, 176);
		_camera.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, 144);
    	
//    	Log.i(Util.TAG, "setupCamera(" + width + ", " + height+")");
//    	synchronized (this){
//    		if(_camera != null && _camera.isOpened()){
//    			// hope for the best
//    			_frameWidth = width;
//    			_frameHeight = height;
//    			
//    			// get all supported preview sizes
//    			final List<Size> possiblePreviewSizes = _camera.getSupportedPreviewSizes();
//    			Log.i(Util.TAG, possiblePreviewSizes.toString());
//    			
//    			// select OPTIMAL preview size
//    			double minDiff = Double.MAX_VALUE;
//    			for(Size size : possiblePreviewSizes){
//    				if(height >= size.height && width > size.width){
//	    				final double currentDiff = Math.max(height - size.height, width - size.width);
//	    				if(currentDiff < minDiff){
//	    					_frameWidth = (int)size.width;
//	    					_frameHeight = (int)size.height;
//	    					minDiff = currentDiff;
//	    				}
//    				}
//    			}
//    			
//    			_camera.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, _frameWidth);
//    			_camera.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, _frameHeight);
//    		}
//    	}
    }
    
    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        Log.i(Util.TAG, "surfaceChanged");
        setupCamera(width, height);
    }

    public void surfaceCreated(SurfaceHolder holder) {
        Log.i(Util.TAG, "surfaceCreated");
        
        // START the processing THREAD !
        (new Thread(this)).start();
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i(Util.TAG, "surfaceDestroyed");
        releaseCamera();
    }


    public void run() {
        Log.i(Util.TAG, "Starting processing thread");
        _fpsMeter.init();
        
        int frames = 0;
        while (frames <= 760) {
            Bitmap bmp = null;
            Mat frame = new Mat();

            synchronized (this) {
                if (_camera == null || !_camera.read(frame)){
                	Log.e(Util.TAG, _camera == null ? "Camera is null " : "Can't grab the camera" + ", probably app has been paused.");
                	
                	// do not break, for when we come back from sleep !
                	Thread.yield();
                	try {
						Thread.sleep(100);
					} catch (InterruptedException e) {
						Log.e(Util.TAG, "", e);
					}
                }else{
	                // to be implemented by the subclass
	                bmp = processFrame(frame);
	
	                _fpsMeter.measure();
                }
            	
            	
//            	if(frames == 1){
//                    // Enable TRACING, AFTER the init()
//                    Debug.startMethodTracing("OpenTLD");
//            	}
//            	
//            	// for DEBUGging
//            	frame = Highgui.imread("/sdcard/TLDtest/test_" + frames++ + ".png"); // add option O for Greyscale
//            	Log.i(Util.TAG, "_____________++++ FRAME : " + (frames-1) + " Size: " + frame.size());
//            	bmp = processFrame(frame);

            }

            if (bmp != null) {
                final Canvas canvas = _holder.lockCanvas();
                if (canvas != null) {
                	_canvasImgXOffset = (canvas.getWidth() - bmp.getWidth()) / 2;
                	_canvasImgYOffset = (canvas.getHeight() - bmp.getHeight()) / 2;
                    
                    canvas.drawBitmap(bmp, _canvasImgXOffset, _canvasImgYOffset, null);
                    
                    
                    _fpsMeter.draw(canvas, _canvasImgXOffset, _canvasImgYOffset);
                    _holder.unlockCanvasAndPost(canvas);
                }
                bmp.recycle();
            }
        }
        
//        // Stop TRACING
//        Debug.stopMethodTracing();
    }
    
    
    
    public int getFrameWidth(){
    	return _frameWidth;
    }
    public int getFrameHeight(){
    	return _frameHeight;
    }
}