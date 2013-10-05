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

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicReference;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.View;

import com.trandi.opentld.tld.BoundingBox;
import com.trandi.opentld.tld.Tld;
import com.trandi.opentld.tld.Tld.ProcessFrameStruct;
import com.trandi.opentld.tld.Util;


public class TLDView extends JavaCameraView implements CameraBridgeViewBase.CvCameraViewListener {
	final private SurfaceHolder _holder;
    private int _canvasImgYOffset;
    private int _canvasImgXOffset;
	
	private Mat _currentGray = new Mat();
	private Mat _lastGray = new Mat();
	private Tld _tld = null;
	private Rect _trackedBox = null;
	private ProcessFrameStruct _processFrameStruct = null;
	private Properties _tldProperties;
	
	public TLDView(Context context, AttributeSet attrs) {
		super(context, attrs);
		_holder = getHolder();
		
		// Init the PROPERTIES
		InputStream propsIS = null;
		try{
			propsIS = context.getResources().openRawResource(R.raw.parameters);
			_tldProperties = new Properties();
			_tldProperties.load(propsIS);
		} catch (IOException e) {
			Log.e(Util.TAG, "Can't load properties", e);
		}finally{
			if(propsIS != null){
				try {
					propsIS.close();
				} catch (IOException e) {
					Log.e(Util.TAG, "Can't close props", e);
				}
			}
		}
		
		// listens to its own events
		setCvCameraViewListener(this);
		
		
		// DEBUG
		//_trackedBox = new BoundingBox(165,93,51,54, 0, 0);
		
		// LISTEN for touches of the screen, to define the BOX to be tracked
		final AtomicReference<Point> trackedBox1stCorner = new AtomicReference<Point>();
		final Paint rectPaint = new Paint();
		rectPaint.setColor(Color.rgb(0, 255, 0));
		rectPaint.setStrokeWidth(5);
		rectPaint.setStyle(Style.STROKE);
		
		setOnTouchListener(new OnTouchListener() {
			@Override
			public boolean onTouch(View v, MotionEvent event) {
				if(_trackedBox == null){
					final Point corner = new Point(event.getX() - _canvasImgXOffset, event.getY() - _canvasImgYOffset);
					switch(event.getAction()){
					case MotionEvent.ACTION_DOWN:
						trackedBox1stCorner.set(corner);
						Log.i(Util.TAG, "1st corner: " + corner);
						break;
					case MotionEvent.ACTION_UP:
						_trackedBox = new BoundingBox(trackedBox1stCorner.get(), corner);
						Log.i(Util.TAG, "Tracked box DEFINED: " + _trackedBox);
						break;
					case MotionEvent.ACTION_MOVE:
						final Canvas canvas =_holder.lockCanvas();
						final android.graphics.Rect rect = new android.graphics.Rect(
										(int)trackedBox1stCorner.get().x + _canvasImgXOffset, (int)trackedBox1stCorner.get().y + _canvasImgYOffset, 
										(int)corner.x + _canvasImgXOffset, (int)corner.y + _canvasImgYOffset);
						canvas.drawRect(rect, rectPaint);
						_holder.unlockCanvasAndPost(canvas);
						break;
					}
				}
				return true;
			}
		});
	}

	@Override
	public Mat onCameraFrame(Mat frame) {
		//Imgproc.resize(frame, frame, new Size(44, 36));
		
		if(_trackedBox != null){
			if(_tld == null){ // run the 1st time only
				Imgproc.cvtColor(frame, _lastGray, Imgproc.COLOR_RGB2GRAY);
				_tld = new Tld(_tldProperties);
				_tld.init(_lastGray, _trackedBox);
			}else{
				Imgproc.cvtColor(frame, _currentGray, Imgproc.COLOR_RGB2GRAY);
			
				_processFrameStruct = _tld.processFrame(_lastGray, _currentGray);
				drawPoints(frame, _processFrameStruct.lastPoints, new Scalar(255, 0, 0));
				drawPoints(frame, _processFrameStruct.currentPoints, new Scalar(0, 255, 0));
				drawBox(frame, _processFrameStruct.currentBBox, new Scalar(0, 0, 255));
					
				_currentGray.copyTo(_lastGray);
			}
		}

        return frame;
	}

	
	@Override
	public void onCameraViewStarted(int width, int height) {
    	_canvasImgXOffset = (getWidth() - width) / 2;
    	_canvasImgYOffset = (getHeight() - height) / 2;
	}

	@Override
	public void onCameraViewStopped() {
		// TODO Auto-generated method stub
	}
	
	
	private static void drawPoints(Mat image, final Point[] points, final Scalar colour){
		if(points != null){
			for(Point point : points){
				Core.circle(image, point, 2, colour);
			}
		}
	}
	
	private static void drawBox(Mat image, final BoundingBox box, final Scalar colour){
		if(box != null){
			Core.rectangle(image, box.tl(), box.br(), colour);
		}
	}	
}
