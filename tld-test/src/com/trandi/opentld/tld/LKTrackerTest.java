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
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

import android.util.Log;

import com.trandi.opentld.tld.Util.Pair;

public class LKTrackerTest extends OpenCVTestCase{
	private static final double [][] LAST_POINTS_IN = new double[][]{{165, 93}, {186, 93}, {207, 93}, {228, 93}, {249, 93}, {270, 93}, {291, 93}, {312, 93}, {333, 93}, {354, 93}, {375, 93}, {165, 107}, {186, 107}, {207, 107}, {228, 107}, {249, 107}, {270, 107}, {291, 107}, {312, 107}, {333, 107}, {354, 107}, {375, 107}, {165, 121}, {186, 121}, {207, 121}, {228, 121}, {249, 121}, {270, 121}, {291, 121}, {312, 121}, {333, 121}, {354, 121}, {375, 121}, {165, 135}, {186, 135}, {207, 135}, {228, 135}, {249, 135}, {270, 135}, {291, 135}, {312, 135}, {333, 135}, {354, 135}, {375, 135}, {165, 149}, {186, 149}, {207, 149}, {228, 149}, {249, 149}, {270, 149}, {291, 149}, {312, 149}, {333, 149}, {354, 149}, {375, 149}, {165, 163}, {186, 163}, {207, 163}, {228, 163}, {249, 163}, {270, 163}, {291, 163}, {312, 163}, {333, 163}, {354, 163}, {375, 163}, {165, 177}, {186, 177}, {207, 177}, {228, 177}, {249, 177}, {270, 177}, {291, 177}, {312, 177}, {333, 177}, {354, 177}, {375, 177}, {165, 191}, {186, 191}, {207, 191}, {228, 191}, {249, 191}, {270, 191}, {291, 191}, {312, 191}, {333, 191}, {354, 191}, {375, 191}, {165, 205}, {186, 205}, {207, 205}, {228, 205}, {249, 205}, {270, 205}, {291, 205}, {312, 205}, {333, 205}, {354, 205}, {375, 205}, {165, 219}, {186, 219}, {207, 219}, {228, 219}, {249, 219}, {270, 219}, {291, 219}, {312, 219}, {333, 219}, {354, 219}, {375, 219}, {165, 233}, {186, 233}, {207, 233}, {228, 233}, {249, 233}, {270, 233}, {291, 233}, {312, 233}, {333, 233}, {354, 233}, {375, 233}};
	private static final double[][] LAST_POINTS_OUT = new double[][]{{186.0, 93.0}, {291.0, 93.0}, {228.0, 121.0}, {312.0, 121.0}, {165.0, 135.0}, {207.0, 135.0}, {312.0, 135.0}, {228.0, 149.0}, {270.0, 149.0}, {291.0, 149.0}, {312.0, 149.0}, {207.0, 163.0}, {312.0, 163.0}, {165.0, 177.0}, {207.0, 177.0}, {249.0, 177.0}, {270.0, 177.0}, {291.0, 177.0}, {165.0, 191.0}, {207.0, 191.0}, {228.0, 191.0}, {249.0, 191.0}, {270.0, 191.0}, {291.0, 191.0}, {165.0, 205.0}, {186.0, 205.0}, {165.0, 219.0}, {186.0, 219.0}, {207.0, 219.0}, {228.0, 219.0}, {312.0, 219.0}};
	private static final double [][] CURR_POINTS = new double[][]{{183.86514282226563, 93.55406188964844}, {291.14886474609375, 100.68718719482422}, {226.94879150390625, 121.74057006835938}, {310.65032958984375, 121.42726135253906}, {162.57179260253906, 135.29327392578125}, {207.0057830810547, 134.99842834472656}, {309.7173156738281, 134.33848571777344}, {228.00534057617188, 148.99209594726563}, {269.8294677734375, 148.4680633544922}, {291.8500061035156, 147.63864135742188}, {310.7330627441406, 149.5430450439453}, {206.3493194580078, 165.28530883789063}, {309.9791259765625, 162.94647216796875}, {165.558837890625, 176.97889709472656}, {206.78158569335938, 177.51510620117188}, {246.7549591064453, 175.75962829589844}, {274.6474609375, 179.42388916015625}, {296.5911560058594, 178.86265563964844}, {164.9466552734375, 191.55120849609375}, {206.85003662109375, 191.38241577148438}, {227.9065704345703, 191.2167510986328}, {244.27536010742188, 190.4522247314453}, {268.47705078125, 191.3587646484375}, {299.8081970214844, 194.59164428710938}, {165.43556213378906, 208.11083984375}, {185.843017578125, 205.44772338867188}, {164.3252716064453, 219.3542022705078}, {186.7926483154297, 218.0596160888672}, {209.62525939941406, 217.36581420898438}, {227.74319458007813, 218.33566284179688}, {311.57666015625, 218.7012176513672}};



	/**
	 * This does NOT give the same exact results as the C++ version on a PC.
	 * The very calcOpticalFlowPyrLK, called with the exact same params returns different set of points with different values !
	 */
	public void testTrack() {
		Mat img1 = readMatFromFile("track_frame_1");
		Mat img2 = readMatFromFile("track_frame_2");
		
		Imgproc.cvtColor(img1, img1, Imgproc.COLOR_RGB2GRAY);
		Imgproc.cvtColor(img2, img2, Imgproc.COLOR_RGB2GRAY);

		Pair<Point[], Point[]> result = new LKTracker().track(img1, img2, toPoints(LAST_POINTS_IN));
Log.i(UtilTest.TAG, Arrays.asList(result.first).toString());
Log.i(UtilTest.TAG, Arrays.asList(result.second).toString());
		assertNotNull("The tracking is broken, can't even return a non null result !", result);
		for(int i=0; i<result.first.length; i++){
			assertEquals("Last points OUT, different for index: " + i, new Point(LAST_POINTS_OUT[i]), result.first[i]);
			assertEquals("Current points, different for index: " + i, new Point(CURR_POINTS[i]), result.second[i]);
		}
	}
	
	
	private static Point[] toPoints(double[][] coordinates){
		final Point[] result = new Point[coordinates.length];
		for(int i=0; i<result.length; i++){
			result[i] = new Point(coordinates[i]);
		}
		return result;
	}

	private static List<Point> toList(Point[] points){
		List<Point> result = new ArrayList<Point>();
		for(Point p : points){
			result.add(p);
		}
		return result;
	}
}
