package com.example.phimua;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Color;
import android.os.Bundle;

//package com.example.regression;
// Link to this app:
// https://medium.com/analytics-vidhya/running-ml-models-in-android-using-tensorflow-lite-e549209287f0
// convert and load model:
// https://margaretmz.medium.com/e2e-tfkeras-tflite-android-273acde6588
// run a single inference:
//https://firebase.google.com/docs/ml/android/use-custom-models

import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.Manifest;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.content.pm.PackageManager;
import android.content.Intent;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "Main Activity";

    static TextView outp;
    Button start_recording;
    Button close_app;
    Button stop_recording;
    final int REQUEST_PERMISSION_CODE =0;
    boolean bound;
    static boolean stopThread = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        start_recording = findViewById(R.id.start_recording);
        close_app = findViewById(R.id.close_app);
        stop_recording = findViewById(R.id.stop_recording);
        outp = findViewById(R.id.hw);
        outp.setTextColor(Color.WHITE);
    }

    public void OnStartBtnClicked(View v){
        if (!checkPermissonfromDevice())
            requestPermission();
        outp.setTextColor(Color.WHITE);
        outp.setText("Running");
        //Intent startServiceIntent = new Intent(getApplicationContext(), ForegroundService.class); // application context
        Intent startServiceIntent = new Intent(this, ForegroundService.class); // activity context

        startService(startServiceIntent);
        start_recording.setEnabled(false);

    }

    public void OnCloseAppBtnClicked(View v){
        finish();
        System.exit(0);
    }
    public void OnStopBtnClicked(View v){
        Intent stopServiceIntent = new Intent(this, ForegroundService.class);
        stopService(stopServiceIntent);
        stopThread = true;
        // TODO: if reset is pressed twice in a row, it displays "resetting..." forever
        outp.setText("Resetting...");
        start_recording.setEnabled(true);
    }

    private void requestPermission() {
        ActivityCompat.requestPermissions(this, new String[]{
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.RECORD_AUDIO
        }, REQUEST_PERMISSION_CODE);
    }

    private boolean checkPermissonfromDevice() {
        int write_external_storage = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int record_audio = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO);
        return (write_external_storage == PackageManager.PERMISSION_GRANTED) && (record_audio == PackageManager.PERMISSION_GRANTED);
    }

}