package com.example.phimua;

import static com.example.phimua.App.CHANNEL_ID;
import static com.example.phimua.MainActivity.outp;
import static com.example.phimua.MainActivity.stopThread;

import static java.lang.Math.abs;
import android.os.Environment;
import android.app.Notification;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.BatteryManager;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.SystemClock;
import android.util.Log;
import android.widget.ProgressBar;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;

import java.util.Date;
import java.util.Timer;
import java.util.TimerTask;

import javax.annotation.Nullable;


public class ForegroundService extends Service {
    private static final String TAG = "Service";
    private static final String FILE_NAME = "audio";

    MediaRecorder myRecorder;
    final int audioSampleRateHigh = 16000;
    final int REQUEST_PERMISSION_CODE = 0;
    File RootFolder, AudioFile;
    public static final int BYTES_PER_SAMPLE = 2;   // 2 for AudioFormat.ENCODING_PCM_16BIT, 1 for ENCODING_PCM_8BIT
    private static final DecimalFormat df = new DecimalFormat("0.00");

    SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss");

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int StartId) {
        Log.d(TAG, "inside ForegroundService onStartCommand()");
        stopThread = false;
        Intent notificationIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent =
                PendingIntent.getActivity(this, 0, notificationIntent,
                        PendingIntent.FLAG_IMMUTABLE);

        Notification notification =
                new Notification.Builder(this, CHANNEL_ID)
                        .setContentTitle("PhIMU-A service")
                        .setContentText("recording audio and IMU data")
                        .setSmallIcon(R.drawable.ic_launcher_foreground)
                        .setContentIntent(pendingIntent)
                        .build();

        // Notification ID cannot be 0.
        startForeground(1, notification);

        //Log.d(TAG, "running foreground service");
        AudioRunnable runnable = new AudioRunnable(10);
        new Thread(runnable).start();

        return START_NOT_STICKY;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "inside ForegroundService onDestroy()");
    }

    private File getOutputMediaFile() {
        Date date = new Date();

        File mediaStorageDir = new File(this.getExternalFilesDir(null)
                + "/sensor_recordings");

        // create storage directory if it does not exist
        // on watch, conversation log file will be saved at: /storage/emulated/0/Android/data/com.example.myapplication/files/conversation_logs/
        if (!mediaStorageDir.exists()) {
            if (!mediaStorageDir.mkdirs()) {
                return null;
            }
        }
        File mediaFile;
        String mFileName = sdf.format(date) + "_audio_record.3gpp";
        mediaFile = new File(mediaStorageDir.getPath() + File.separator + mFileName);
        return mediaFile;
    }

    class AudioRunnable implements Runnable {
        int seconds;
        int min_buffer_size_high = AudioRecord.getMinBufferSize(audioSampleRateHigh,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT);

        AudioRecord audioRecordHigh = new AudioRecord(MediaRecorder.AudioSource.MIC,
                audioSampleRateHigh,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                min_buffer_size_high);

        AudioRunnable(int seconds) {
            this.seconds = seconds;
        }

        @Override
        public void run() {
            SimpleDateFormat s = new SimpleDateFormat("yyyyMMdd_hhmmss");

            //String outputFile;
            File outputFile = getOutputMediaFile();

            myRecorder = new MediaRecorder();
            myRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            myRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
            myRecorder.setAudioEncoder(MediaRecorder.OutputFormat.AMR_NB); // 4.75 to 12.2 kbps sampled @ 8kHz
            // media formats: https://developer.android.com/guide/topics/media/media-formats
            myRecorder.setOutputFile(outputFile);

            try {
                myRecorder.prepare();
                myRecorder.start();
            } catch (IllegalStateException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }

            Timer myTimer;

            myTimer = new Timer();
            myTimer.schedule(new TimerTask() {
                @Override
                public void run() {
                    try {
                        myRecorder.stop();
                        myRecorder.reset();
                        myRecorder.release();
                        myRecorder = null;

                        stopSelf(); // ??
                    } catch (IllegalStateException e) {
                        //  it is called before start()
                        e.printStackTrace();
                    } catch (RuntimeException e) {
                        // no valid audio/video data has been received
                        e.printStackTrace();
                    }
                }
            }, 60000 * 5); //5 minutes timer to stop the recording after 90 minutes

        //Log.d(TAG, "thread ending");
        //outp.post(new Runnable() {
        //    @Override
        //    public void run() {
        //        outp.setTextColor(Color.WHITE);
        //        outp.setText("Recording ended");
        //    }
        //});
    }
}
/*
    class IMURunnable implements Runnable {
        int seconds;
        int min_buffer_size_high = AudioRecord.getMinBufferSize(audioSampleRateHigh,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT);

        AudioRecord audioRecordHigh = new AudioRecord(MediaRecorder.AudioSource.MIC,
                audioSampleRateHigh,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                min_buffer_size_high);

        IMURunnable(int seconds) {
            this.seconds = seconds;
        }

        @Override
        public void run() {
            FileOutputStream fos_phone = null;

            Date date = new Date();

            byte[] bufferHigh = new byte[BYTES_PER_SAMPLE * 1 * audioSampleRateHigh * 30]; // 30s audio at 16kHz

            try {
                // on phone, conversation log file will be saved at: /Android/data/data/com.example.myapplication/files/conversation_logs/
                fos_phone = openFileOutput(sdf.format(date) + "_" + FILE_NAME, MODE_PRIVATE);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }

            do {
                Date date2 = new Date();
                Log.d(TAG, "start audio recording");
                Log.d(TAG,"16kHz sampling");
                audioRecordHigh.startRecording();
                audioRecordHigh.read(bufferHigh, 0, bufferHigh.length);
                //Arrays.fill(bufferHigh, (byte) 0); // uncomment this line and audio.start()/read()/release() to skip audio recording
                float[] audio_in_float = convertByteToFloat(bufferHigh);
                Log.d("min buffer size", Integer.toString(min_buffer_size_high));
                Log.d("audio data size", Integer.toString(audio_in_float.length));
                audioRecordHigh.stop();

                // write results to terminal log
                Log.d(TAG, "Sensor: ");

                // write results to log file
                String text = "Time: " + sdf.format(date2)
                        + "\n";
                try {
                    fos_phone.write(text.getBytes());
                } catch (IOException e) {
                    e.printStackTrace();
                }
                // display results on UI
                // final variables for thread below

                outp.post(new Runnable() {
                    @Override
                    public void run() {
                        outp.setTextColor(Color.WHITE);
                        outp.setText("Time: " + sdf.format(date2));
                    }
                });

                //wakeLock.release();
            } while (!stopThread);

            outp.post(new Runnable() {
                @Override
                public void run() {
                    outp.setText("Run stopped");
                }
            });
            audioRecordHigh.release();
            Log.d(TAG, "16kHz mic released");
            Log.d(TAG, "stopping thread");
            return; // stopThread must be true for this line to run
        }
    }
*/
    /*
    private void setupMediaRecorder() {
        mediaRecorder = new MediaRecorder();
        mediaRecorder.release();
    }*/

    private static float[] convertByteToFloat(byte[] bytes) {
        int bytesPerSample = BYTES_PER_SAMPLE;
        int numberOfSamples = bytes.length / bytesPerSample;

        float[] floats = new float[numberOfSamples];
        for (int i = 0; i < numberOfSamples; i++) {
            for (int b = 0; b < bytesPerSample; b++) {
                int v = bytes[bytesPerSample * i + b];
                if (b < bytesPerSample - 1 || bytesPerSample == 1) {
                    v &= 0xFF; // TODO: ??
                }
                floats[i] += v << (b * 8);
            }
            floats[i] /= 32768f;
        }
        return floats;
    }

    public static int getBatteryPercentage(Context context) {
        BatteryManager bm = (BatteryManager) context.getSystemService(BATTERY_SERVICE);
        return bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
    }
}