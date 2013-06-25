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

import java.text.DecimalFormat;

import org.opencv.core.Core;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

public class FpsMeter {
	private final static int TEXT_SIZE = 40;
	
    final int                   step = 20;
    final double                freq = Core.getTickFrequency();
    final DecimalFormat         decimalFormat = new DecimalFormat("0.00");
    
    int                         framesCounter;
    long                        prevFrameTime;
    String                      strfps;
    Paint                       paint;

    public void init() {
        framesCounter = 0;
        prevFrameTime = Core.getTickCount();
        strfps = "";

        paint = new Paint();
        paint.setColor(Color.BLUE);
        paint.setTextSize(TEXT_SIZE);
    }

    public void measure() {
        framesCounter++;
        if (framesCounter % step == 0) {
            final long time = Core.getTickCount();
            final double fps = step * freq / (time - prevFrameTime);
            prevFrameTime = time;
            strfps = decimalFormat.format(fps) + " FPS";
        }
    }

    public String getFps(){
    	return strfps;
    }
    
    public void draw(Canvas canvas, float offsetx, float offsety) {
        canvas.drawText(strfps, offsetx, TEXT_SIZE + offsety, paint);
    }

}
