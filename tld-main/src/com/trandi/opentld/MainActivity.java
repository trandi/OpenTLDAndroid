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

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.Window;
import android.widget.LinearLayout;

import com.trandi.opentld.tld.Util;

public class MainActivity extends Activity {
	private TLDView _tldView;
    
    private BaseLoaderCallback _openCVCallBack = new BaseLoaderCallback(this) {
    	@Override
    	public void onManagerConnected(int status) {
    		switch (status) {
				case LoaderCallbackInterface.SUCCESS:
				{
					Log.i(Util.TAG, "OpenCV loaded successfully");

					// Create and set View (replace the bogus one created by the main.xml layout)
					_tldView = new TLDView(mAppContext);
					((LinearLayout)findViewById(R.id.cameraFeedHolder)).addView(_tldView);
					
					// Check native OpenCV camera
					if( !_tldView.openCamera() ) {
						AlertDialog ad = new AlertDialog.Builder(mAppContext).create();
						ad.setCancelable(false); // This blocks the 'BACK' button
						ad.setMessage("Fatal error: can't open camera!");
						ad.setButton("OK", new DialogInterface.OnClickListener() {
						    public void onClick(DialogInterface dialog, int which) {
						    	dialog.dismiss();
						    	finish();
						    }
						});
						ad.show();
					}
				} break;
				default:
				{
					super.onManagerConnected(status);
				} break;
			}
    	}
    };	
	
	
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_main);
        
        
        Log.i(Util.TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, _openCVCallBack)) {
        	Log.e(Util.TAG, "Cannot connect to OpenCV Manager");
        }
    }    

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.activity_main, menu);
        return true;
    }
}
