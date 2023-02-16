## Set up on macOS
1. Download Android Studio
2. Download Android Studio File Transfer (needed to connect with phone)
3. Install ADB for macOS.   
If you already have Android Studio installed, add these lines in your `.bash_rc` or `.bash_profile`.
```
export ANDROID_HOME=/Users/$USER/Library/Android/sdk
export PATH=${PATH}:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools
```
Source the file to refresh the terminal.
```
source ~/.bash_profile
```
4. Clone this repo.

## Set up on Phone
1. Enable developer options on phone and allow permissions for your computer to access files on phone. 
2. Connect the phone to the computer using a USB cable.
3. Run `adb devices` in the terminal to verify that the phone is connected.
```bash-3.2$ adb devices
List of devices attached
c9126f49	device
```
4. Android Studio should detect the connected phone. Run the app in Android Studio and the app should appear on the phone.

## Set up on Watch (Fossil DW10F1 i.e. Carlyle HR 0344)
1. Factory reset watch
2. Download WearOS app on phone.
3. Pair watch with phone through WearOS app. If on a local wifi network the watch should now be connected to the same wifi network as your phone.
4. Enable developer options on the watch.

### Wifi debugging 
1. Enable wifi-debug on the watch. Once this is selected, it should display the watch's IP address.
2. In a mac terminal, run `adb connect <ip address>` to connect the watch to the ADB debugger. This is the IP address of the watch displayed on the watch screen under Enable Wifi-Debug. Run `adb connect` again to verify the connection. The outputs should appear as below.
```
$ bash-3.2$ adb connect 192.168.1.91:5555
failed to authenticate to 192.168.1.91:5555
$ bash-3.2$ adb connect 192.168.1.91:5555
already connected to 192.168.1.91:5555
```
3. Open this project in Android Studio and run the app on the Carlyle HR device. The app should now appear on the watch face.

### Bluetooth debugging
1. Enable bluetooth debug on the watch
2. Set up the phone as outlined above. 
3. Ensure that running `adb devices` shows the phone as an available device.
4. Follow steps here: https://developer.android.com/training/wearables/get-started/debugging#bt-debugging
    - Ensure that under Developer Options on the phone, USB configuration allows transferrring of files.
    - Once the watch is connected via the steps above, `adb devices` should show the watch and phone as available devices. If the phone does not appear as an available device, the phone may need to be directly connected to the laptop (i.e. not through a USB hub).
    ```
    bash-3.2$ adb devices
    List of devices attached
    R38MA0ETJAF	device
    127.0.0.1:4444	device
    ```
5. In Android Studio, the watch should be an available device to select from the devices drop down menu. If the device does not appear, debug the steps above. 
6. Click Run in Android Studio to run the app on the watch.


### Save data off watch
View files saved on watch at: `/data/data/com.example.myapplication/files/`.  
Copy files to local machine with:  
```
adb -s 127.0.0.1:4444 pull /storage/emulated/0/Android/data/com.example.myapplication/files/conversation_logs/20230123_133931_conversation_record.txt .
```
