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

import org.opencv.core.Rect;



public class GridTest extends OpenCVTestCase {
	
	public void testGrid(){
		final Grid grid = new Grid(getTestMat(), new Rect(165, 93, 51, 54), 15);
		assertEquals(58901, grid.getSize());
		
		grid.updateGoodBadBoxes(10);
		assertEquals(10, grid.getGoodBoxes().length);
		assertEquals(57855, grid.getBadBoxes().length);
		assertEquals(new BoundingBox(166, 91, 51, 54, 0.8940853f, 6), grid.getBestBox());
		assertEquals(new BoundingBox(157,86,67,70,-1,-1), grid.getBBhull());
		
		assertEquals(new BoundingBox(157, 91, 61, 65, 0.6945776f, 7), grid.getGoodBoxes()[0]);
		assertEquals(new BoundingBox(163, 91, 61, 65, 0.6945776f, 7), grid.getGoodBoxes()[1]);
		assertEquals(new BoundingBox(171, 96, 51, 54, 0.71428573f, 6), grid.getGoodBoxes()[2]);
		assertEquals(new BoundingBox(166, 101, 51, 54, 0.7169576f, 6), grid.getGoodBoxes()[3]);
		assertEquals(new BoundingBox(171, 91, 51, 54, 0.7386364f, 6), grid.getGoodBoxes()[4]);
		assertEquals(new BoundingBox(166, 86, 51, 54, 0.7441419f, 6), grid.getGoodBoxes()[5]);
		assertEquals(new BoundingBox(161, 96, 51, 54, 0.7704918f, 6), grid.getGoodBoxes()[6]);
		assertEquals(new BoundingBox(161, 91, 51, 54, 0.79765016f, 6), grid.getGoodBoxes()[7]);
		assertEquals(new BoundingBox(166, 96, 51, 54, 0.86206895f, 6), grid.getGoodBoxes()[8]);
		assertEquals(new BoundingBox(166, 91, 51, 54, 0.8940853f, 6), grid.getGoodBoxes()[9]);
	}
}
