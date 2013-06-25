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
import org.opencv.core.Point;
import org.opencv.core.Rect;

import android.util.Log;


public class BoundingBox extends Rect{
	private static final int POINTS_MAX_COUNT = 10;
	private static final int POINTS_MARGIN_H = 0;
	private static final int POINTS_MARGIN_V = 0;
	
	public float overlap = -1;	// overlap with current(main) BoundingBox
	int scaleIdx = -1;

	
	public BoundingBox(){
		super();
	}
	
	public BoundingBox(final Point ul, final Point br){
		super(ul, br);
	}
	
	public BoundingBox(int x, int y, int width, int height, float overlap, int scaleIdx){
		super(x, y, width, height);
		this.overlap = overlap;
		this.scaleIdx = scaleIdx;
	}
	
	float calcOverlap(final Rect other){
		if(x > other.x + other.width || y > other.y + other.height || x + width < other.x || y + height < other.y){
			// obvious case where these 2 boxes do not overlap at all !
			return 0f;
		}else{
			final float colIntersection =  Math.min(x + width, other.x + other.width) - Math.max(x, other.x);
			final float rowIntersection =  Math.min(y + height, other.y + other.height) - Math.max(y, other.y);
			
			final float intersection = colIntersection * rowIntersection;
			final float myArea = width * height;
			final float otherArea = other.width * other.height;
			
			return intersection / (myArea + otherArea - intersection);			
		}
	}
	
	
	Point[] points(){
		final List<Point> result = new ArrayList<Point>();
		final int stepx = (int) Math.ceil((width - 2 * POINTS_MARGIN_H) / POINTS_MAX_COUNT);
		final int stepy = (int) Math.ceil((height - 2 * POINTS_MARGIN_V) / POINTS_MAX_COUNT);	
		for(int j = y + POINTS_MARGIN_V; j < y + height - POINTS_MARGIN_V; j += stepy){
			for(int i = x + POINTS_MARGIN_H; i < x + width - POINTS_MARGIN_H; i += stepx){
				result.add(new Point(i, j));
			}
		}
		Log.i(Util.TAG, "Points in BB: " + this + " stepx=" + stepx + " stepy=" + stepy + " RES size=" + result.size());			
		return result.toArray(new Point[result.size()]);
	}
	
	
	BoundingBox predict(final Point[] points1, final Point[] points2){
		if(points1.length != points2.length) throw new IllegalArgumentException("The 2 arrays of points must be of the same lenght ! (" + points1.length + ", " + points2.length + ")");
		
		final int npoints = points1.length;
		Log.i(Util.TAG, "Tracked points: " + npoints);
		
		final float[] xoff = new float[npoints];
		final float[] yoff = new float[npoints];
		for(int i = 0; i < npoints; i++){
			xoff[i] = (float) (points2[i].x - points1[i].x);
			yoff[i] = (float) (points2[i].y - points1[i].y);
		}
		final float dx = Util.median(xoff);
		final float dy = Util.median(yoff);
		
		float s = 1f;
		if(npoints > 1){
			final float[] d = new float[npoints * (npoints - 1) / 2];
			int idx = 0;
			for(int i = 0; i < npoints; i++){
				for(int j = i + 1; j < npoints; j++){
					d[idx++] = Util.norm(points2[i], points2[j]) / Util.norm(points1[i], points1[j]);
				}
			}
			s = Util.median(d);
		}
		
		final float s1 = 0.5f * (s - 1) * width;
		final float s2 = 0.5f * (s - 1) * height;
		final BoundingBox result = new BoundingBox();
		result.x = Math.round(x + dx - s1);
		result.y = Math.round(y + dy - s2);
		result.width = Math.round(width * s);
		result.height = Math.round(height * s);
		
		Log.i(Util.TAG, "Current BB: " + this + ", Predicted BB: " + result);
		
		return result;
	}
	
	
	BoundingBox intersect(final Mat img){
		final BoundingBox result = new BoundingBox();
		result.x = Math.max(x, 0);
		result.y = Math.max(y, 0);
		result.width = (int) Math.min(Math.min(img.cols() - x, width), Math.min(width, br().x));
		result.height = (int) Math.min(Math.min(img.rows() - y, height), Math.min(height, br().y));
		return result;
	}
	
	@Override
	public String toString(){
		return "(" + x + ", " + y + ", " + width + ", " + height + " / " + overlap + ", " + scaleIdx + ")";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = super.hashCode();
		result = prime * result + Float.floatToIntBits(overlap);
		result = prime * result + scaleIdx;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (!super.equals(obj))
			return false;
		if (getClass() != obj.getClass())
			return false;
		BoundingBox other = (BoundingBox) obj;
		if (Float.floatToIntBits(overlap) != Float
				.floatToIntBits(other.overlap))
			return false;
		if (scaleIdx != other.scaleIdx)
			return false;
		return true;
	}
}
